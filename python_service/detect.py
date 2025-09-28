import sys
from landingai.pipeline.image_source import VideoFile
from landingai.predict import Predictor, OcrPredictor
from landingai import visualize
from landingai.postprocess import crop

video_path = sys.argv[1]
video_source = VideoFile(video_path, samples_per_second=1)
frames = [f.image for f in video_source]

predictor = Predictor("e001c156-5de0-43f3-9991-f19699b31202", api_key={Your API Key Here})
ocr = OcrPredictor(api_key="{Your API Key Here}")

bounding_boxes = [predictor.predict(frame) for frame in frames]
cropped_imgs = [crop(bboxes, frame) for frame, bboxes in zip(frames, bounding_boxes)]

for cropped_set in cropped_imgs:
  for plate in cropped_set:
    ocr_pred = ocr.predict(plate)
    for text in ocr_pred:
      print(text.text)
