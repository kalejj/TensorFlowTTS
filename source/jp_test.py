import os
import sys
from datetime import datetime
import tensorflow as tf
import time
import yaml
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy as np
import re
# from g2pk import G2p
# from G2p.trans import sentranslit as trans
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

import scipy.io.wavfile as wavf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class VoiceSynthesis:
    # 모델 초기화 
    def __init__(self):
        # # gpu memory의 1/3 만을 할당하기로 제한
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.200) 
        conf = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        # 탄력적인 메모리 할당
        # conf.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=conf)
        module_path = os.path.dirname(os.path.abspath(__file__))

        # fastspeech2
        fastspeech2_config = AutoConfig.from_pretrained(os.path.join(module_path,'examples/fastspeech2/conf/fastspeech2.jsut.v1.yaml'))
        self.fastspeech2 = TFAutoModel.from_pretrained(
            config=fastspeech2_config,
            pretrained_path=os.path.join(module_path, 'model-100000.h5'),
            name="fastspeech2"
        )

        # mb-melgan
        mb_melgan_config = AutoConfig.from_pretrained(os.path.join(module_path,'examples/multiband_melgan/conf/multiband_melgan.baker.v1.yaml'))
        self.mb_melgan = TFAutoModel.from_pretrained(
            config=mb_melgan_config,
            pretrained_path=os.path.join(module_path, 'generator-1000000.h5'),
            name="mb_melgan"
        )
        
        #processor - 글자 별 상응하는 숫자의 mapper 설정 가져오기
        self.processor = AutoProcessor.from_pretrained(pretrained_path=os.path.join(module_path,"tensorflow_tts/processor/pretrained/jsut_mapper.json"))

    def do_synthesis(self, input_ids, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
        
        # text2mel part
        if text2mel_name == "TACOTRON":
            start = time.time()
            _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
                tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                tf.convert_to_tensor([len(input_ids)], tf.int32),
                tf.convert_to_tensor([0], dtype=tf.int32)
            )
            print('taco_time:', time.time() - start)
        elif text2mel_name == "FASTSPEECH2":
            mel_before, mel_outputs, duration_outputs, _, _ = text2mel_model.inference(
                tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
                speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            )
        else:
            raise ValueError("Only TACOTRON, FASTSPEECH2 are supported on text2mel_name")

        # vocoder part
        if vocoder_name == "MB-MELGAN" or vocoder_name == "hifigan":
            audio = vocoder_model.inference(mel_outputs)[0, :, 0]
        else:
            raise ValueError("Only MB_MELGAN, hifigan are supported on vocoder_name")

        if text2mel_name == "TACOTRON":
            return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
        else:
            return mel_outputs.numpy(), audio.numpy()

    # text to speech method - 이걸 호출해서 인자 문구를 넣고 사용. 생성한 목소리 파일 경로 반환
    def text_to_voice(self,input_text):
        
        # 입력 문장을 ktml_tag, pause_tag, 문장([. ? !]) 단위로 split
        text_list = re.split("\. |\? |! |<[ktml|pause]+>[^<>]*</[ktml|pause]+>", input_text)
        split = re.findall('\. |\? |! |<[ktml|pause]+>[^<>]*</[ktml|pause]+>', input_text)

        for i in range(len(split)):
            text_list[i] = text_list[i].strip() + split[i].strip()
        print('text:', text_list)

        # TTS 변환
        total_audios = np.array([0], dtype = 'float32')
        for i in range(len(text_list)):
            # ktml Tag
            if re.findall('<ktml>[^<>]*</ktml>', text_list[i]) != []:
                text_list[i] = text_list[i].replace('<ktml>', '').replace('</ktml>', '')
                text_list[i] = text_list[i].strip()
                total_audios = np.append(total_audios, librosa.load(f'ktml/{text_list[i]}.wav', sr = 22050)[0], axis = 0)
                text_list[i] = f'효과음: {text_list[i]}.wav'

            # pause tag
            elif re.findall('<pause>[^<>]*</pause>', text_list[i]) != []:
                text_list[i] = text_list[i].replace('<pause>', '').replace('</pause>', '')
                text_list[i] = text_list[i].strip()
                total_audios = np.append(total_audios, np.array([0]*round(22050*float(text_list[i])), dtype = 'float32'), axis = 0)
                text_list[i] = f'공백: {text_list[i]}초'

            # tag 없음
            else:
                # text_list[i] = re.sub("\((.+?)\)|\[(.+?)\]|\{(.+?)\}|\<(.+?)\>", "", text_list[i])
                # text_list[i] = trans(text_list[i])
                # print('text:', text_list[i])
                # g2p = G2p()
                # text_list[i] = g2p(text_list[i])
                input_ids = self.processor.text_to_sequence(text_list[i], inference = True)
                # input_ids= self.processor.text_to_sequence(text_list[i])
                # mels, alignment_history, audios = self.do_synthesis(input_ids, self.tacotron2, self.mb_melgan, "TACOTRON", "MB-MELGAN")
                # mels, alignment_history, audios = self.do_synthesis(input_ids, self.tacotron2, self.hifigan, "TACOTRON", "hifigan")
                mels, audios = self.do_synthesis(input_ids, self.fastspeech2, self.mb_melgan, "FASTSPEECH2", "MB-MELGAN")
                # mels, audios = self.do_synthesis(input_ids, self.fastspeech2, self.hifigan, "FASTSPEECH2", "hifigan")
                total_audios = np.append(total_audios, audios, axis = 0)

            total_audios = np.append(total_audios, np.array([0]*7000, dtype = 'float32'))
        print('preprocessed_text:', text_list)

        # 현재시간을 파일 제목으로 사용
        cur_time = datetime.now()
        timestamp_str = cur_time.strftime("%Y%m%d_%H%M%S_%f")
        sample_rate = 22050
        # audio가 저장될 위치 - ./output/
        
        output_audio = os.path.join(os.path.dirname(os.path.abspath(__file__)),'output', timestamp_str +'.wav')
        # wavf.write(output_audio, sample_rate, audios)
        sf.write(output_audio, total_audios, sample_rate)
        return output_audio



if __name__ == "__main__":

    tts = VoiceSynthesis()
    start = time.time()  # 시작 시간 저장
    input_text = "新入社員は、初めての失敗を、うまくいいぬけた。"
    print(tts.text_to_voice(input_text))
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간