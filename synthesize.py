from logging import raiseExceptions
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
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

import scipy.io.wavfile as wavf

# 합성시 사용할 GPU DEVICE선택 (-1: CPU, 0: 0번GPU, 1: 1번GPU ...)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(os.getcwd())
class VoiceSynthesis:
    # 모델 초기화 
    def __init__(self):
        module_path = os.path.dirname(os.path.abspath(__file__))
        
        # tacotron2
        tacotron2_config = AutoConfig.from_pretrained(os.path.join(module_path,'examples/tacotron2/conf/tacotron2.kss.v1.yaml'))
        self.tacotron2 = TFAutoModel.from_pretrained(
            config=tacotron2_config,
            pretrained_path=os.path.join(module_path,"tacotron2-200k.h5"),
            name="tacotron2"
        )

        # fastspeech2
        fastspeech2_config = AutoConfig.from_pretrained(os.path.join(module_path,'examples/fastspeech2/conf/fastspeech2.kss.v1.yaml'))
        self.fastspeech2 = TFAutoModel.from_pretrained(
            config=fastspeech2_config,
            pretrained_path=os.path.join(module_path, 'fastspeech2-200k.h5'),
            name="fastspeech2"
        )

        # mb-melgan
        mb_melgan_config = AutoConfig.from_pretrained(os.path.join(module_path,'examples/multiband_melgan/conf/multiband_melgan.v1.yaml'))
        self.mb_melgan = TFAutoModel.from_pretrained(
            config=mb_melgan_config,
            pretrained_path=os.path.join(module_path, 'MB-MelGAN-1300k.h5'),
            name="mb_melgan"
        )
        
        #processor - 글자 별 상응하는 숫자의 mapper 설정 가져오기
        self.processor = AutoProcessor.from_pretrained(pretrained_path=os.path.join(module_path,"tensorflow_tts/processor/pretrained/kss_mapper.json"))

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
        if vocoder_name == "MB-MELGAN":
            audio = vocoder_model.inference(mel_outputs)[0, :, 0]
        else:
            raise ValueError("Only MB_MELGAN is supported on vocoder_name")

        if text2mel_name == "TACOTRON":
            return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
        else:
            return mel_outputs.numpy(), audio.numpy()

    # text to speech method - 이걸 호출해서 인자 문구를 넣고 사용. 생성한 목소리 파일 경로 반환
    def text_to_voice(self,input_text, model_name):
        digit_dict = {"0": "공","1": "일","2": "이","3": "삼","4": "사","5": "오","6": "육","7": "칠","8": "팔","9": "구"}
        email_dict = {"0": "공", "1": "일", "2": "이", "3": "삼", "4": "사", "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구", "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "쥐", "H": "에이치", "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피", "Q": "큐", "R": "알", "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유", "X": "엑스", "Y": "와이", "Z": "지", "@": " 골뱅이 ", ".": " 쩜 "}
        
        # 입력 문장을 effect_tag, pause_tag, 문장([. ? !]) 단위로 split
        origin = re.split("\. |\? |! |<[effect|pause]+>[^<>]*</[effect|pause]+>", input_text)
        split = re.findall('\. |\? |! |<[effect|pause]+>[^<>]*</[effect|pause]+>', input_text)
        text_list = []
        print(origin)
        print(split)
        for i in range(len(split)):
            if (split[i] == '. ') or (split[i] == '? ') or (split[i] == '! '):
                a = origin[i] + split[i]
                text_list.append(a.strip())
            else:
                text_list.append(origin[i])
                text_list.append(split[i])
        text_list.append(origin[-1])
        print('text:', text_list)
        
        # digit, email tag 한글로 변환
        for x in range(len(text_list)):
            # digit_tag
            while '<digit>' in text_list[x]:
                start = text_list[x].find('<digit>')
                end = text_list[x].find('</digit>') + 8
                new = text_list[x][start:end].replace('<digit>', '').replace('</digit>', '')
                new = re.sub('[^0-9]', ' ', new)
                for i, j in digit_dict.items():
                    new = new.replace(i, j)
                text_list[x] = text_list[x].replace(text_list[x][start:end], ' ' + new)
                text_list[x] = text_list[x].replace('  ', ' ')
            # email_tag
            while '<email>' in text_list[x]:
                start = text_list[x].find('<email>')
                end = text_list[x].find('</email>') + 8
                new = text_list[x][start:end].replace('<email>', '').replace('</email>', '')
                new = new.upper()
                for i, j in email_dict.items():
                    new = new.replace(i, j)
                text_list[x] = text_list[x].replace(text_list[x][start:end], ' ' + new + ' ')
                text_list[x] = text_list[x].replace('  ', ' ')

        # TTS 변환
        total_audios = np.array([0], dtype = 'float32')
        print(text_list)
        for i in range(len(text_list)):
            # effect Tag
            if re.findall('<effect>[^<>]*</effect>', text_list[i]) != []:
                text_list[i] = text_list[i].replace('<effect>', '').replace('</effect>', '')
                text_list[i] = text_list[i].strip()
                total_audios = np.append(total_audios, librosa.load(f'effect/{text_list[i]}.wav', sr = 22050)[0], axis = 0)
                text_list[i] = f'효과음: {text_list[i]}.wav'

            # pause tag
            elif re.findall('<pause>[^<>]*</pause>', text_list[i]) != []:
                text_list[i] = text_list[i].replace('<pause>', '').replace('</pause>', '')
                text_list[i] = text_list[i].strip()
                total_audios = np.append(total_audios, np.array([0]*round(22050*float(text_list[i])), dtype = 'float32'), axis = 0)
                text_list[i] = f'공백: {text_list[i]}초'

            # tag 없음
            else:
                if (text_list[i][-1] != '.') & (text_list[i][-1] != '?') & (text_list[i][-1] != '!'):
                    text_list[i] = text_list[i] + '.'
                input_ids, text1 = self.processor.text_to_sequence(text_list[i])
                text_list[i] = ''.join(text1)
                if model_name == 'tacotron2':
                    mels, alignment_history, audios = self.do_synthesis(input_ids, self.tacotron2, self.mb_melgan, "TACOTRON", "MB-MELGAN")
                elif model_name == 'fastspeech2':
                    mels, audios = self.do_synthesis(input_ids, self.fastspeech2, self.mb_melgan, "FASTSPEECH2", "MB-MELGAN")
                else:
                    raise ValueError("Only tacotron2, fastspeech2 are supported on model_name")
                total_audios = np.append(total_audios, audios, axis = 0)
            total_audios = np.append(total_audios, np.array([0]*7000, dtype = 'float32'))
        print('preprocessed_text:', text_list)

        # 현재시간을 파일 제목으로 사용
        cur_time = datetime.now()
        timestamp_str = cur_time.strftime("%Y%m%d_%H%M%S_%f")
        sample_rate = 22050
        
        # audio가 저장될 위치 - ./output/
        if model_name =='tacotron2':
            output_audio = os.path.join(os.path.dirname(os.path.abspath(__file__)),'output', 'tacotron2', timestamp_str +'.wav')
        elif model_name =='fastspeech2':
            output_audio = os.path.join(os.path.dirname(os.path.abspath(__file__)),'output', 'fastspeech2', timestamp_str +'.wav')
        else:
            raise ValueError("Only tacotron2, fastspeech2 are supported on model_name")
        sf.write(output_audio, total_audios, sample_rate)
        return output_audio



if __name__ == "__main__":

    tts = VoiceSynthesis()
    start = time.time()  # 시작 시간 저장
    input_text = "들의 콩깍지는 깐 콩깍지인가 안깐 콩깍지인가. 깐 콩깍지면 어떻고 안 깐 콩각지면 어떠냐. 깐 콩깍지나 안 깐 콩깍지나 콩깍지는 다 콩깍지인데. 상표 붙인 큰 깡통은 깐 깡통인가? 안 깐 깡통인가? 앞 집 팥죽은 붉은 팥 풋팥죽이고, 뒷집 콩죽은 햇콩단콩 콩죽, 우리집 깨죽은 검은깨 깨죽인데 사람들은 햇콩 단콩 콩죽 깨죽 죽먹기를 싫어하더라."
    print(tts.text_to_voice(input_text, 'fastspeech2'))
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간