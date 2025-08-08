import os,sys,subprocess
import json
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(CURRENT_PATH)

import qwenaudio.hubert.inference as hubert
import qwenaudio.whisper_svc.inference as whisper_svc

def process(wav_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    path_vec = os.path.join(out_path, "vec.npy")
    path_pit = os.path.join(out_path, "pit.npy")
    path_ppg = os.path.join(out_path, "ppg.npy")
    if not os.path.exists(path_ppg):
        print(
            f"Auto run : python whisper_svc/inference.py -w {wav_path} -p {path_ppg}"
        )
        # os.system(f"python whisper_svc/inference.py -w {args.wave} -p {args.ppg}")
        subprocess.run([sys.executable, os.path.join(ROOT_PATH, "src", "qwenaudio", "whisper_svc", "inference.py"), "-w", wav_path, "-p", path_ppg])

    if not os.path.exists(path_vec):
        print(f"Auto run : python hubert/inference.py -w {wav_path} -v {path_vec}")
        # os.system(f"python hubert/inference.py -w {args.wave} -v {args.vec}")
        subprocess.run([sys.executable, os.path.join(ROOT_PATH, "src", "qwenaudio", "hubert", "inference.py"), "-w", wav_path, "-v", path_vec])

    if not os.path.exists(path_pit):
        print(f"Auto run : python pitch/inference.py -w {wav_path} -p {path_pit}")
        # os.system(f"python pitch/inference.py -w {args.wave} -p {args.pit}")
        subprocess.run([sys.executable, os.path.join(ROOT_PATH, "src", "qwenaudio", "pitch", "inference.py"), "-w", wav_path, "-p", path_pit])

if __name__ == "__main__":

    gen_json_path = "/home/w-4090/projects/qwenaudio/data/gen2.json"
    score_json_path = "/home/w-4090/projects/qwenaudio/data/scores.json"
    data = []
    
    with open(gen_json_path, 'r') as f:
        gen_json = json.load(f)
    
    with open(score_json_path, 'r') as f:
        score_json = json.load(f)
    
    process_list = []
    
    for song_id, song_data in score_json.items():
        for audio_id, audio_body in song_data.items():
            audio_path = audio_body["audio_path"].replace("cutted_score_audio", "cutted_score_audio_separated")

            if audio_id in gen_json:

                rc_id = audio_path.split("/")[-2]+"-"+audio_id

                gen_text_list = gen_json[audio_id]
                print(audio_path, rc_id)
                # process_list.append((audio_path, rc_id))
                process(audio_path, "/home/w-4090/projects/qwenaudio/data/audio_feat/"+rc_id)