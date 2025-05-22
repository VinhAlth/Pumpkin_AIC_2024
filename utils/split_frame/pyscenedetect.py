from scenedetect import open_video, SceneManager, split_video_ffmpeg, scene_manager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
import os
import pathlib
import pandas as pd

dataset_0 = "/dataset/AIC2024/original_dataset/0/videos/"
dataset_1 = "/dataset/AIC2024/original_dataset/1/videos/"
dataset_2 = "/dataset/AIC2024/original_dataset/2/videos/"

#pyscenedetect_t_5
keyframe_0_pyscene_t_5 ="/dataset/AIC2024/pumkin_dataset/0/frames/pyscenedetect_t_5_fix/"


def split_video_into_scenes(video_path, save_path, video_name, threshold=10, encoder_param=95):
    if (os.path.exists(str(save_path + video_name).replace(".mp4",".csv"))):
        return
    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=5))
    manager.detect_scenes(video, show_progress=True)
    scene_list = manager.get_scene_list()
    #split_video_ffmpeg(video_path, scene_list, show_progress=True)
    frame_list = []
    for i, scene in enumerate(scene_list):
        print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
            i+1,
            scene[0].get_timecode(), scene[0].get_frames(),
            scene[1].get_timecode(), scene[1].get_frames(),))
        scene_name = "scene" + str(i+1)
        cur_tuple = (scene_name, scene[0].get_timecode(), scene[1].get_timecode(),
                     scene[0].get_frames(), scene[1].get_frames())
        frame_list.append(cur_tuple)
    
    df = pd.DataFrame(frame_list, columns=['scene name', 
                                           'start_time', 'end_time', 
                                           'start_frame', 'end_frame'])
    df.to_csv(str(save_path + video_name).replace(".mp4",".csv"), index=False)
    #num_images = the number of images to snatch from each scene
    scene_manager.save_images(scene_list = scene_list, 
                              video = video,
                              output_dir = save_path,
                              num_images=2,
                              image_name_template="$FRAME_NUMBER",
                              encoder_param=encoder_param)


# video_path = "/dataset/AIC2023/0/Videos_L01/video/L01_V001.mp4"
# save_path = "/workspace/competitions/AIC_2023/SIU_Pumpkin/samples/pyscenedetect/L01_V001/"

# split_video_into_scenes(video_path=video_path, save_path=save_path)


def get_frames(video_path, video_name, saveframe_path, threshold=5, encoder_param=95):
    print(video_path)
    path = pathlib.Path(saveframe_path)
    path.mkdir(parents=True, exist_ok=True)
    split_video_into_scenes(video_path, saveframe_path, video_name, threshold=threshold, encoder_param=encoder_param)
    print("FINISHED")

def splitframe(dataset_path, keyframe_path, threshold=5, encoder_param=95):
    files = [f for f in os.listdir(dataset_path) if str(f).startswith("Videos")]
    for f in files:
        video_path = dataset_path + f + "/video/"
        print(video_path)
        for video in os.listdir(video_path):
            # if "L04_V029" not in video:
            #      continue
            print(video)
            saved_directory = keyframe_path + "Keyframes_" + video[:3] + "/keyframes/" + video.replace(".mp4","") + "/"
            cur_path = video_path + video
            get_frames(cur_path, video, saved_directory, threshold=threshold, encoder_param=encoder_param)
            #print(get_fps(cur_path))

splitframe(dataset_0, keyframe_0_pyscene_t_5, 5, 100)