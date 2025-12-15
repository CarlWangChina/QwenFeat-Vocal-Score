import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import audioscore.model

if __name__=="__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = audioscore.model.SongEvalGenerator_audio_lora()
    model.load_model(os.path.join(ROOT_DIR,"ckpts", "SongEvalGenerator", "step_2_al_audio", "best_model_step_132000"))
    model = model.half() #可选
    model = model.cuda()

    score = model.generate_tag("/data/nfs/audioscore/data/sort/data/audio/203887_250822105518005501.m4a")

    print("score:", score)
