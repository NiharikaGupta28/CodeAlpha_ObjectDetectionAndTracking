from datetime import datetime
import cv2
import time

def run_video_pipeline(source=0, save_output=False):
    """
    source: 0 for webcam, or a filename string like 'test_video.mp4' for a video file
    save_output: if True, saves the processed video to output.mp4
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open source: {source}")
        return

    # Get video properties (needed if we want to save output later)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)

    writer = None
    output_filename = None
    if save_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_filename, fourcc, fps_input if fps_input > 0 else 20, (frame_width, frame_height))
    # ---- NEW: create a resizable window ----
    cv2.namedWindow("Video Pipeline", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Pipeline", 480, 854)  # scaled-down portrait size that fits most screens

    prev_time = 0  # used to calculate FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        # ---- FPS calculation ----
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Overlay FPS text on the frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Video Pipeline", frame)

        if save_output and writer is not None:
            writer.write(frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved output as: {output_filename}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Change source to 0 for webcam, or "test_video.mp4" for the video file
    run_video_pipeline(source="test_video.mp4", save_output=False)