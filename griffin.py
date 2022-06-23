import glob
import tempfile
import time
import librosa.display
import yaml
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import GridBox, Label, Layout, Audio
from tensorflow_tts.utils import TFGriffinLim, griffin_lim_lb
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
        module_path = os.path.dirname(os.path.abspath(__file__))
        
        tacotron2_config = AutoConfig.from_pretrained(os.path.join(module_path,'examples/tacotron2/conf/tacotron2.kss.v1.yaml'))
        self.tacotron2 = TFAutoModel.from_pretrained(
            config=tacotron2_config,
            pretrained_path=os.path.join(module_path,"model_out/male_tacotron2/checkpoints/model-5000.h5"),
            name="tacotron2"
        )
        
        # fastspeech2_config = AutoConfig.from_pretrained(os.path.join(module_path,'examples/fastspeech2/conf/fastspeech2.kss.v1.yaml'))
        # self.fastspeech2 = TFAutoModel.from_pretrained(
        #     config=fastspeech2_config,
        #     pretrained_path=os.path.join(module_path, 'model_out/fastspeech2/checkpoints/model-200000.h5'),
        #     name="fastspeech2"
        # )
        
        self.processor = AutoProcessor.from_pretrained(pretrained_path=os.path.join(module_path,"test/files/kss_mapper.json"))
    
    def do_synthesis(self, input_ids, text2mel_model, text2mel_name):
        
        # text2mel part
        if text2mel_name == "TACOTRON":
            start = time.time()
            _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
                tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                tf.convert_to_tensor([len(input_ids)], tf.int32),
                tf.convert_to_tensor([0], dtype=tf.int32)
            )
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

        return mel_outputs.numpy()
        
    def text_to_voice(self,input_text):
        digit_dict = {"0": "공","1": "일","2": "이","3": "삼","4": "사","5": "오","6": "육","7": "칠","8": "팔","9": "구"}
        email_dict = {"0": "공, ", "1": "일, ", "2": "이, ", "3": "삼, ", "4": "사, ", "5": "오, ", "6": "육, ", "7": "칠, ", "8": "팔, ", "9": "구, ", "A": "에이, ", "B": "비, ", "C": "씨, ", "D": "디, ", "E": "이, ", "F": "에프, ", "G": "쥐, ", "H": "에이치, ", "I": "아이, ", "J": "제이, ", "K": "케이, ", "L": "엘, ", "M": "엠, ", "N": "엔, ", "O": "오, ", "P": "피, ", "Q": "큐, ", "R": "알, ", "S": "에스, ", "T": "티, ", "U": "유, ", "V": "브이, ", "W": "더블유, ", "X": "엑스, ", "Y": "와이, ", "Z": "지, ", "@": "골뱅이, ", ".": "쩜, "}
        
        # 입력 문장을 ktml_tag, pause_tag, 문장([. ? !]) 단위로 split
        text_list = re.split("\. |\? |! |<[ktml|pause]+>[^<>]*</[ktml|pause]+>", input_text)
        split = re.findall('\. |\? |! |<[ktml|pause]+>[^<>]*</[ktml|pause]+>', input_text)

        for i in range(len(split)):
            text_list[i] = text_list[i].strip() + split[i].strip()
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
                text_list[x] = text_list[x].replace(text_list[x][start:end], ' ' + new + ' ')
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
        # print('2:', text_list)
        
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
                input_ids, text1 = self.processor.text_to_sequence(text_list[i])
                mel_outputs = self.do_synthesis(input_ids, self.tacotron2, "TACOTRON") 
                # mel_outputs = self.do_synthesis(input_ids, self.fastspeech2, "FASTSPEECH2")
                stats_path = "dump_male/stats.npy"
                dataset_config_path = "preprocess/kss_preprocess.yaml"
                config = yaml.load(open(dataset_config_path), Loader=yaml.Loader)
                s = mel_outputs.shape
                mel_outputs = mel_outputs.reshape((s[1], s[2]))
                inv_wav_lb = griffin_lim_lb(mel_outputs, stats_path, config)  # [mel_len] -> [audio_len]

                lb_wav = tf.audio.encode_wav(inv_wav_lb[:, tf.newaxis], config["sampling_rate"])
                
                # sf.write('gl_lib.wav', inv_wav_lb, 22050)
                # mels, audios = self.do_synthesis(input_ids, self.fastspeech2, "FASTSPEECH2")
                total_audios = np.append(total_audios, inv_wav_lb, axis = 0)

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
    # input_text = "17일 게임업계에 따르면 지난 13일 라인게임즈가 서비스를 시작한 니즈게임즈 개발 신작 멀티플랫폼 핵앤슬래시 액션 역할수행게임 언디셈버가 구글 플레이스토어에서 전체 인기 게임 1위에 올랐다. 이 게임은 열세번째 존재 서펜스의 부활을 막기 위한 룬 헌터의 스토리를 기반으로, 시나리오 모드와 이용자간 대결 콘텐츠인 영광의 성전, 몬스터 대결, 협동 콘텐츠, 레이드, 성장의 한계를 체험해볼 수 있는 카오스 던전 등 다양한 엔드 콘텐츠로 구성돼 있는 것이 특징이다."
    # input_text = "퓨렌스는 매년 송년회, 우수사원 시상, 전사 권고연차를 실시하였습니다. 그러나 2021년은 아래와 같은 상황을 고려하여 부득이하게 송년회행사와 전사 권고연차를 진행하지 않기로 하였습니다. 우수사원 시상은 경영지원팀에서 개별적으로 찾아뵙고 진행예정입니다. 첫번째, 현재 진행중인 우리은행, 신한생명등에 프로젝트 참여인력은 연말 상관없이 근무예정. 두번째, 코비드 19 상황 고려. 열심히 근무하고 있는 임직원 분들의 노고에 감사드리며 경영지원팀에서도 지원을 아끼지 않겠습니다. 궁금하신 사항은 경영지원팀에 문의하여 주세요. 감사합니다."
    # input_text = "이 카페에는 젊은 연인들이 많이 와요."
    input_text = "신한생명과 오렌지라이프의 통합법인인 신한라이프는, 1일 오전, 서울 중구 본사에서 출범식을 개최했다고 밝혔다. 이날 행사에는 조용병 신한금융그룹회장과, 그룹사 최고경영자, 성대규 신한라이프 사장, 신입사원을 포함한 임직원 대표들이 참석했다."
    # input_text = "웹서비스, 클라우드, IT솔루션 등 광범위한 피해발생 가능성이 높습니다."
    # input_text = "내가 그린 구름그림은 새털구름 그린 구름그림이고, 네가 그린 구름그림은 깃털구름 그린 구름그림이다."
    print(tts.text_to_voice(input_text))
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간