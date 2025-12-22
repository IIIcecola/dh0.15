# --------------------
# 处理视频以及音频
# --------------------


from moviepy.editor import VideoFileClip
import os

# -----------------------------
#   功能 1：从 MP4 中提取 WAV
# -----------------------------
def extract_audio_from_mp4(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path, codec='pcm_s16le')
    print(f"音频已成功提取到: {output_audio_path}")


# -----------------------------
#   功能 2：按时间间隔切分 MP4
# -----------------------------
def split_video_by_interval(video_path, interval, output_dir):
    video = VideoFileClip(video_path)
    duration = video.duration

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = 0
    index = 1

    while start_time < duration:
        end_time = min(start_time + interval, duration)
        subclip = video.subclip(start_time, end_time)

        out_path = os.path.join(output_dir, f"segment_{index}.mp4")
        subclip.write_videofile(out_path, codec="libx264", audio_codec="aac")

        print(f"✔ 保存分段视频: {out_path}")
        start_time = end_time
        index += 1

    print("视频分段完成！")


# 批量视频分割
def VideoSplitMain():
    ''' '''

# 批量音频提取
def AudioExtractMain():
    ''' '''

if __name__ == '__main__':
    ''' '''


