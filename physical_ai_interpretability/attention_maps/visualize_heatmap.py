import io
import cv2
import torch
import queue
import base64
import uvicorn
import argparse
import threading

display_queue = queue.Queue(maxsize=2)

def display_worker():
    """描画専用のバックグラウンドスレッド"""
    cv2.namedWindow("Attention Heatmap", cv2.WINDOW_NORMAL)
    while True:
        try:
            # キューから画像を取得（タイムアウトを入れて定期的にwaitKeyが呼ばれるようにする）
            img = display_queue.get(timeout=0.03)
            cv2.imshow("Attention Heatmap", img)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Display error: {e}")
        
        # OpenCVのウィンドウを更新するために必須
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break