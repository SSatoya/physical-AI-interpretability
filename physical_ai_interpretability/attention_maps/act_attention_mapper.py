import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Optional

class ACTPolicyWithAttention:
    """
    EN：
    Wrapper for ACTPolicy that provides transformer attention visualizations.

    JP:
    ACTPolicyのラッパーで、Transformerの注意マップの可視化機能を提供します。
    """
    
    def __init__(self, policy, preprocessor, image_shapes=None, specific_decoder_token_index: Optional[int] = None):
        """
        EN:
        Initialize the wrapper with an ACTPolicy.
        
        Args:
            policy: An instance of ACTPolicy
            preprocessor: The preprocessor function to apply to observations before passing to the policy
            image_shapes: Optional list of image shapes [(H1, W1), (H2, W2), ...] if known in advance
            specific_decoder_token_index: experimental, allows visualising attention maps for a particular token rather than averaging all outputs.

        JP:
        ACTPolicyのラッパーを初期化します。

        Args:
            policy: ACTPolicyのインスタンス
            preprocessor: 観測をポリシーに渡す前に適用する前処理関数
            image_shapes: 事前にわかっている場合の画像の形状のリスト [(H1, W1), (H2, W2), ...]
            specific_decoder_token_index: 実験的な機能で、すべての出力を平均するのではなく、特定のトークンの注意マップを可視化することを可能にします。
        """
        self.policy = policy  # ベースとなるACTPolicyを保存
        self.preprocessor = preprocessor  # 
        self.config = policy.config
        
        self.specific_decoder_token_index = specific_decoder_token_index
        if self.specific_decoder_token_index is not None:
            if not hasattr(self.config, 'chunk_size'):
                raise AttributeError("Policy's config object does not have 'chunk_size' attribute.")
            if not (0 <= self.specific_decoder_token_index < self.config.chunk_size):
                raise ValueError(
                    f"specific_decoder_token_index ({self.specific_decoder_token_index}) "
                    f"must be between 0 and chunk_size-1 ({self.config.chunk_size - 1})."
                )

        # Determine number of images from config
        # 画像の数をconfigから決定する。image_featuresが指定されていない場合は0とする。
        if self.config.image_features:
            self.num_images = len(self.config.image_features)
        else:
            self.num_images = 0
            
        # Store image shapes if provided, otherwise will be detected at runtime
        # 画像の形状を提供された場合は保存し、そうでない場合はランタイムで検出します。
        self.image_shapes = image_shapes
        
        # For storing the last processed images and attention
        # 最後に処理した画像と注意マップを保存するための変数
        self.last_observation = None
        self.last_attention_maps = None

        if not hasattr(self.policy, 'model') or \
        not hasattr(self.policy.model, 'decoder') or \
        not hasattr(self.policy.model.decoder, 'layers') or \
        not self.policy.model.decoder.layers:
            raise AttributeError("Policy model structure does not match expected ACT architecture for target_layer.")
        
        # ここで最後のTransformer層のMultiheadAttentionモジュールをターゲットにしている。モデル構造によってはここを調整する必要がある。
        self.target_layer = self.policy.model.decoder.layers[-1].multihead_attn  

    def _get_image_backbone(self):
        """
        EN: Return the image backbone used by the wrapped ACT model.
        JP: ラップされたACTモデルで使用されている画像バックボーンを返します。
        """
        model = self.policy.model
        if hasattr(model, "backbone"):
            return model.backbone  # LeRobotのACTの画像処理部分`lerobot/src/lerobot/policies/act/modeling_act.py`
        if hasattr(model, "vision_encoder") and hasattr(model.vision_encoder, "resnet_feature_extractor"):
            return model.vision_encoder.resnet_feature_extractor
        raise AttributeError(
            "Policy model does not expose an image backbone. Expected 'backbone' or 'vision_encoder.resnet_feature_extractor'."
        )
        
    def select_action(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[np.ndarray]]:
        """
        EN:
        Extends policy.select_action to also compute attention maps.
        
        Args:
            observation: Dictionary of observations
            
        Returns:
            action: The predicted action tensor
            attention_maps: List of attention maps, one for each image
        
        JP:
        policy.select_action を拡張し、アテンションマップも計算

        Args:
            observation: 観測の辞書
        
        Returns:
            action: 予測されたアクションのテンソル
            attention_maps: 画像ごとのアテンションマップのリスト
        """
        # Store the observation for later use
        # 後で使用するために観測を保存
        self.last_observation = observation.copy()
        
        # Process the images through the backbone first to understand spatial dimensions
        # 空間次元を理解するために、まずバックボーンを通して画像を処理する
        # NOTE: バックボーンとは？：ACTモデルの中で画像を処理して特徴マップを生成する部分のこと。通常はResNetなどのCNNが使われる。
        images = self._extract_images(observation)                     # 観測辞書から画像テンソルを抽出する
        image_spatial_shapes = self._get_image_spatial_shapes(images)  # CNNに通した画像の形状(15, 20) x 画像分のリスト[(15, 20), (15, 20), (15, 20)]を取得する
        
        # Set up hook to capture attention weights
        # 注意重みをキャプチャするためのフックを設定
        attention_weights_capture = []
        
        def attention_hook(module, input_args, output_tuple):
            """
            EN:
            Capture the attention weights
            In some MultiheadAttention implementations, the attention weights
            might be returned with shape: [batch_size, tgt_len, src_len]
            or [batch_size, num_heads, tgt_len, src_len]

            JP:
            Attention weightsをキャプチャする
            一部のMultiheadAttentionの実装では、注意重みは次の形状で返されることがあります：[batch_size, tgt_len, src_len]
            または [batch_size, num_heads, tgt_len, src_len]

            Args:
                module: フックが登録されたモジュール（MultiheadAttention）
                input_args: その層に渡された入力データ
                output_tuple: その層からの出力。通常は (output, attn_weights) のタプルで、attn_weightsが注意重みを含む。
            """
            
            if isinstance(output_tuple, tuple) and len(output_tuple) > 1:
                # If output is a tuple with attention weights as second element
                # MultiheadAttentionは(計算結果の特徴量, アテンション重み)のタプルを返す．アテンション重みは，画像やトークンのどこを何％注目しているかを示す．
                attn_weights = output_tuple[1]
                # print(f"Captured attention weights with shape: {attn_weights.shape}")  # 注意重みの形状を出力[1, 100, 902](1, Chunk_size(Queryの数), エンコーダーの出力トークン数)
            else:
                # If output format is different, try to get weights from the module directly
                # Some implementations store attention weights in the module after forward pass
                attn_weights = getattr(module, 'attn_weights', None)
            
            if attn_weights is not None:
                # Store the weights regardless of shape - we'll handle reshape later
                # アテンション重みをリストに保存．
                attention_weights_capture.append(attn_weights.detach().cpu())
        
        # Register the hook
        # NOTE: register_forward_hook:「この層の forward（順伝播の計算）が終わった瞬間に、指定した関数を実行してね」とPyTorchに予約することができる機能
        handle = self.target_layer.register_forward_hook(attention_hook)
        
        # Call the original policy's select_action
        # 受け取ったカデータ(画像やjoint)をテンソルの正規化や次元の調整などを実行
        # その後、元のポリシーのselect_actionを呼び出す。
        # フックが登録されているので、この呼び出しの中でターゲット層の順伝播が行われると、自動的にattention_hookが実行されて注意重みがキャプチャされる。
        observation = self.preprocessor(observation)
        with torch.inference_mode():
            action = self.policy.select_action(observation)
            if isinstance(action, tuple):
                action, _ = action
            self.policy.reset()  # チャンクサイズ分を保持しているが，今回は1ステップ分のアクションしか必要ないため，リセットして次のステップに備える．

        # Remove the hook
        # フックを削除して、次のステップで再度登録できるようにする。
        # フックは一度登録すると、明示的に削除しない限りずっと残ってしまうため、ここで確実に削除する。
        handle.remove()
                
        # Process the attention weights
        if attention_weights_capture:
            attn = attention_weights_capture[0].to(action.device)  # memo:(1, 100, 902)
            attention_maps, proprio_attention = self._map_attention_to_images(attn, image_spatial_shapes)
            self.last_attention_maps = attention_maps
            self.last_proprio_attention = proprio_attention  # Store for visualization
        else:
            print("Warning: No attention weights were captured.")
            attention_maps = [None] * self.num_images
            self.last_attention_maps = attention_maps
            self.last_proprio_attention = 0.0  # Store for visualization
            
        return action, attention_maps

    def _extract_images(self, observation: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        EN: Extract image tensors from observation dictionary
        JP: 観測辞書から画像テンソルを抽出する
        """
        images = []
        for key in self.config.image_features:
            if key in observation:
                images.append(observation[key])
        return images
    
    def _get_image_spatial_shapes(self, images: List[torch.Tensor]) -> List[Tuple[int, int]]:
        """
        EN:
        Get the spatial shapes of the feature maps after ResNet processing.
        For ResNet, this is typically H/32 × W/32

        JP:
        ResNet処理後の特徴マップの空間形状を取得
        ResNetの場合、通常はH/32 × W/32となる
        """
        spatial_shapes = []
        backbone = self._get_image_backbone()  # バックボーン（画像処理部分）を取得
        for img_tensor in images:
            if img_tensor is None:
                spatial_shapes.append((0, 0))
                continue
                
            # Run image through backbone to get feature map shape
            # ここでは、画像テンソルをバックボーンに通して特徴マップを取得し、その空間的な形状を記録
            with torch.no_grad():
                if img_tensor.dim() == 3:
                    img_tensor_batched = img_tensor.unsqueeze(0)
                else:
                    img_tensor_batched = img_tensor

                # img_tensor_batched = img_tensor_batched.to(next(self.policy.model.vision_encoder.resnet_feature_extractor.parameters()).device)
                img_tensor_batched = img_tensor_batched.to(next(self.policy.model.parameters()).device)

                feature_map_dict = backbone(img_tensor_batched)
                feature_map = feature_map_dict["feature_map"]
                h, w = feature_map.shape[2], feature_map.shape[3]
                spatial_shapes.append((h, w))
                # print(f"Extracted spatial shape for image: {h}x{w}")  # 15x20: ResNetの特徴マップの空間的な形状を出力

        return spatial_shapes
    
    def _map_attention_to_images(self,
                                attention: torch.Tensor,
                                image_spatial_shapes: List[Tuple[int, int]]) -> Tuple[List[np.ndarray], float]:
        """
        EN:
        Map transformer attention weights back to the original images and extract proprioception attention.

        Normalizes attention maps globally across all images AND proprioception for this timestep.

        Args:
            attention: Tensor of shape [batch, heads, tgt_len, src_len]
                       (tgt_len is config.chunk_size)
            image_spatial_shapes: List of (height, width) tuples for feature maps

        Returns:
            Tuple of:

            - List of globally normalized attention maps as numpy arrays

            - Proprioception attention value (float, normalized to same scale as visual attention)

        JP:
        トランスフォーマーの注意重みを元の画像にマッピングし，アテンション値を抽出する．
        この関数では、すべての画像とこのタイムステップの固有感覚に対して、注意マップをグローバルに正規化します。

        Args:
            attention: [batch, heads, tgt_len, src_len]の形状のテンソル
                       (tgt_lenはconfig.chunk_size)
            image_spatial_shapes: 特徴マップの (高さ, 幅) のタプルのリスト
        
        Returns:
            タプルで返す：
            - グローバルに正規化された注意マップのリスト（numpy配列）
            - 自己受容感覚（jointへ）の注意値（float、視覚的注意と同じスケールに正規化）
        """
        if attention.dim() == 4:
            attention = attention.mean(dim=1)  # -> [batch, tgt_len, src_len]
        elif attention.dim() != 3:
            raise ValueError(f"Unexpected attention dimension: {attention.shape}. Expected 3 or 4.")

        # Token structure: [latent(潜在変数), (robot_state(ロボットの状態)), (env_state(環境の状態(ないかも？))), (image_tokens(15x20に分割された画像の特徴))
        n_prefix_tokens = 1  # latent token
        proprio_token_idx = None
        if self.config.robot_state_feature:
            proprio_token_idx = n_prefix_tokens  # proprioception is the next token
            n_prefix_tokens += 1
        if self.config.env_state_feature:
            n_prefix_tokens += 1
        # memo：propriod_token_idex：1, n_prefix_tokens: 2


        # --- Step 1: Extract proprioception attention ---
        proprio_attention = 0.0
        if proprio_token_idx is not None:
            # Extract attention to proprioception token
            if self.specific_decoder_token_index is not None:  # default：None
                if 0 <= self.specific_decoder_token_index < attention.shape[1]:
                    proprio_attention_tensor = attention[:, self.specific_decoder_token_index, proprio_token_idx]
                else:
                    proprio_attention_tensor = attention[:, :, proprio_token_idx].mean(dim=1)
            else:
                proprio_attention_tensor = attention[:, :, proprio_token_idx].mean(dim=1)  # joint値への注意を平均して抽出する(attention[:, :, proprio_token_idx]:[batch, 100])

            # Take first batch element
            proprio_attention = proprio_attention_tensor[0].cpu().numpy().item()

        # --- Step 2: Collect all raw (unnormalized) 2D numpy attention maps ---
        raw_numpy_attention_maps = []
        # Store the per-image token counts for reshaping, needed later
        tokens_per_image = [h * w for h, w in image_spatial_shapes]  # [300, 300, 300]など、画像ごとのトークン数のリストを作成する。ResNetの特徴マップの空間的な形状から計算される。


        current_src_token_idx = n_prefix_tokens
        for i, (h_feat, w_feat) in enumerate(image_spatial_shapes):
            if h_feat == 0 or w_feat == 0:
                raw_numpy_attention_maps.append(None)
                if tokens_per_image[i] > 0: # if shape was (0,0) but tokens_per_image[i] was not 0
                    current_src_token_idx += tokens_per_image[i]
                continue

            num_img_tokens = tokens_per_image[i]  # トークンサイズを取り出す（Ex: 300）
            start_idx = current_src_token_idx     # 画像iに対するトークンの開始インデックスを計算する
            end_idx = start_idx + num_img_tokens  # 画像iに対するトークンの終了インデックスを計算する
            current_src_token_idx = end_idx

            attention_to_img_features = attention[:, :, start_idx:end_idx]

            if self.specific_decoder_token_index is not None:  # default：None
                if not (0 <= self.specific_decoder_token_index < attention_to_img_features.shape[1]):
                    print(f"Warning (map_attention): specific_decoder_token_index {self.specific_decoder_token_index} "
                          f"is out of bounds for actual tgt_len {attention_to_img_features.shape[1]}. "
                          f"Falling back to averaging.")
                    img_attn_tensor_for_map = attention_to_img_features.mean(dim=1)
                else:
                    img_attn_tensor_for_map = attention_to_img_features[:, self.specific_decoder_token_index, :]
            else:
                img_attn_tensor_for_map = attention_to_img_features.mean(dim=1)  # 画像iに対するトークンへの注意を平均して1Dテンソルにする（Ex: [batch, 300]）

            # error hadling
            if img_attn_tensor_for_map.shape[0] > 1 and i == 0: # Print once
                 print(f"Warning (map_attention): Batch size is {img_attn_tensor_for_map.shape[0]}. Processing first element for attention map.")

            if img_attn_tensor_for_map.shape[1] != num_img_tokens:
                print(f"Warning (map_attention): Mismatch in token count for image {i}. "
                      f"Expected {num_img_tokens}, got {img_attn_tensor_for_map.shape[1]}. "
                      f"Skipping map for this image.")
                raw_numpy_attention_maps.append(None)
                continue

            try:
                # Get the tensor for the first batch item, still on device
                img_attn_map_1d_tensor = img_attn_tensor_for_map[0] # [num_img_tokens] トークンを取り出す
                # Reshape to 2D tensor
                img_attn_map_2d_tensor = img_attn_map_1d_tensor.reshape(h_feat, w_feat)  # ResNetの特徴マップの空間的な形状に合わせて1Dテンソルを2Dに変形する
                raw_numpy_attention_maps.append(img_attn_map_2d_tensor.cpu().numpy())
            except RuntimeError as e:
                print(f"Error (map_attention): Reshaping attention for image {i}: {e}. "
                      f"Shape was {img_attn_tensor_for_map[0].shape}, target HxW: {h_feat}x{w_feat}. "
                      f"Num tokens: {num_img_tokens}. Skipping.")
                raw_numpy_attention_maps.append(None)
                continue

        # --- Step 3: Find global min and max from all valid raw maps AND proprioception ---
        # すべての有効な生のマップと固有感覚からグローバルな最小値と最大値を見つける（全体の中でどこを一番優先して見ているかを比較できるようにするため）
        global_min = float('inf')
        global_max = float('-inf')
        found_any_valid_map = False

        # Include proprioception attention in global scaling
        if proprio_attention is not None:  # joint
            if proprio_attention < global_min:
                global_min = proprio_attention
            if proprio_attention > global_max:
                global_max = proprio_attention
            found_any_valid_map = True

        for raw_map_np in raw_numpy_attention_maps:
            if raw_map_np is not None:  # camera
                current_min = raw_map_np.min()
                current_max = raw_map_np.max()
                if current_min < global_min:
                    global_min = current_min
                if current_max > global_max:
                    global_max = current_max
                found_any_valid_map = True

        # error handling
        if not found_any_valid_map:
            # All maps were None, return the list of Nones
            return raw_numpy_attention_maps, 0.0
        # If global_min and global_max are still inf/-inf, it means all maps were empty or had issues
        # This case should be covered by found_any_valid_map, but as a safe guard:
        if global_min == float('inf') or global_max == float('-inf'):
            print("Warning (map_attention): Could not determine global min/max for attention. All maps might be invalid.")
            # Fallback: return unnormalized maps or list of Nones
            return [np.zeros_like(m, dtype=np.float32) if m is not None else None for m in raw_numpy_attention_maps], 0.0

        # --- Step 4: Normalize proprioception attention ---
        if global_max > global_min:
            normalized_proprio_attention = (proprio_attention - global_min) / (global_max - global_min)
        else:
            normalized_proprio_attention = 0.0

        # --- Step 5: Normalize all valid visual attention maps using global min/max ---
        final_normalized_attention_maps = []
        for raw_map_np in raw_numpy_attention_maps:
            if raw_map_np is None:
                final_normalized_attention_maps.append(None)
                continue

            if global_max > global_min:
                # Perform normalization
                normalized_map = (raw_map_np - global_min) / (global_max - global_min)
            else:
                # All values across all valid maps are the same (e.g., all are 0.001, or all are 0)
                # Create a uniform map (e.g., all zeros or all 0.5s)
                # If global_max == global_min, it implies all values are equal to global_min (or global_max).
                # If global_min is 0, then (raw_map_np - 0) / (0-0) is problematic.
                # A common practice is to make such a map uniform, often zeros.
                normalized_map = np.zeros_like(raw_map_np, dtype=np.float32)
                # If you prefer a mid-gray for perfectly flat attention:
                # normalized_map = np.full_like(raw_map_np, 0.5, dtype=np.float32)
            final_normalized_attention_maps.append(normalized_map)

        return final_normalized_attention_maps, normalized_proprio_attention
    
    def visualize_attention(self, 
                        images: Optional[List[torch.Tensor]] = None, 
                        attention_maps: Optional[List[np.ndarray]] = None, 
                        observation: Optional[Dict[str, torch.Tensor]] = None,
                        use_rgb: bool = False,
                        overlay_alpha: float = 0.5,
                        show_proprio_border: bool = True,
                        proprio_border_width: int = 15) -> List[np.ndarray]:
        """
        EN:
        Create visualizations by overlaying attention maps on images.
        
        Args:
            images: List of image tensors (optional)
            attention_maps: List of attention maps (optional)
            observation: Observation dict (optional, used if images not provided)
            use_rgb: Whether to use RGB for visualization
            overlay_alpha: Alpha value for attention overlay
            
        Returns:
            List of visualization images as numpy arrays

        JP:
        注意マップを画像に重ねる

        Args:
            images: 画像テンソルのリスト（オプション）
            attention_maps: 注意マップのリスト（オプション）
            observation: 観測辞書
            use_rgb: 可視化にRGBを使用するかどうか
            overlay_alpha: 注意オーバーレイのアルファ値

        Returns:
            可視化画像のリスト（numpy配列）
        """
        # If no images provided, use from observation or last observation
        # 画像が提供されない場合は、観測から、または最後の観測から画像を抽出する
        if images is None:
            if observation is not None:
                images = self._extract_images(observation)
            elif self.last_observation is not None:
                images = self._extract_images(self.last_observation)
            else:
                raise ValueError("No images provided and no stored observation available")
        
        # If no attention maps provided, use last computed ones
        if attention_maps is None:
            if self.last_attention_maps is not None:
                attention_maps = self.last_attention_maps
            else:
                raise ValueError("No attention maps provided and no stored attention maps available")

        # Get proprioception attention value
        proprio_attention = getattr(self, 'last_proprio_attention', 0.0)  # 自己受容感覚の注意値を取得。なければ0.0をデフォルトとする。             
        visualizations = []
        
        for i, (img, attn_map) in enumerate(zip(images, attention_maps)):
            if img is None or attn_map is None:
                visualizations.append(None)
                continue
                
            # Convert tensor to numpy
            # imgがテンソルの場合、NumPy配列に変換する。画像の次元を(H,W,C)に移動して、必要に応じて正規化する。
            if isinstance(img, torch.Tensor):  
                # Move channels to last dimension (H,W,C) for visualization
                if img.dim() == 4:  # (B,C,H,W)
                    img = img.squeeze(0)
                img_np = img.permute(1, 2, 0).cpu().numpy()
                # Normalize if needed
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
            else:
                img_np = img
                
            # Get image dimensions
            h, w = img_np.shape[:2]  # Ex: (height: 480, width: 640)
            
            # Resize attention map to match image size
            attn_map_resized = cv2.resize(attn_map, (w, h))  # Expand:(15, 20) -> (480, 640) 
            
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)  # 0が青、1が赤のカラーマップを適用してヒートマップを作成する
            if use_rgb:
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Create overlay with attention
            vis = cv2.addWeighted(  # 元のカメラ画像とヒートマップを 半透明にして重ね合わせ
                np.uint8(255 * img_np), 1 - overlay_alpha,
                heatmap, overlay_alpha, 0
            )

            # Add proprioception attention border
            if show_proprio_border and proprio_attention > 0:
                # Convert normalized proprioception attention to color intensity
                border_intensity = int(255 * proprio_attention)
                # Create border color (use a different colormap for proprioception)
                # Using magenta/purple to distinguish from visual attention
                if use_rgb:
                    border_color = (border_intensity, 0, border_intensity)  # Magenta in RGB
                else:
                    border_color = (border_intensity, 0, border_intensity)  # Magenta in BGR
                
                # Draw border rectangles (outer and inner rectangles to create border effect)
                # Outer rectangle (full border)
                cv2.rectangle(vis, (0, 0), (w-1, h-1), border_color, proprio_border_width)

                # Optional: Add text label showing proprioception attention value
                text = f"Proprio: {proprio_attention:.3f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # Draw background rectangle for text
                cv2.rectangle(vis, (5, 5), (5 + text_width + 10, 5 + text_height + 10), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(vis, text, (10, 5 + text_height), font, font_scale, (255, 255, 255), thickness)
            
            visualizations.append(vis)
            
        return visualizations
    
    # Forward other methods to the original policy
    def __getattr__(self, name):
        if name not in self.__dict__:
            return getattr(self.policy, name)
        return self.__dict__[name]
