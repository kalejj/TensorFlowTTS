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
            pretrained_path=os.path.join(module_path, 'model-200000.h5'),
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
    def text_to_voice(self,input_text):
        digit_dict = {"0": "공","1": "일","2": "이","3": "삼","4": "사","5": "오","6": "육","7": "칠","8": "팔","9": "구"}
        email_dict = {"0": "공", "1": "일", "2": "이", "3": "삼", "4": "사", "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구", "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "쥐", "H": "에이치", "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피", "Q": "큐", "R": "알", "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유", "X": "엑스", "Y": "와이", "Z": "지", "@": " 골뱅이 ", ".": " 쩜 "}
        
        # 입력 문장을 ktml_tag, pause_tag, 문장([. ? !]) 단위로 split
        origin = re.split("\. |\? |! |<[ktml|pause]+>[^<>]*</[ktml|pause]+>", input_text)
        split = re.findall('\. |\? |! |<[ktml|pause]+>[^<>]*</[ktml|pause]+>', input_text)
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
                if (text_list[i][-1] != '.') & (text_list[i][-1] != '?') & (text_list[i][-1] != '!'):
                    text_list[i] = text_list[i] + '.'
                input_ids, text1 = self.processor.text_to_sequence(text_list[i])
                text_list[i] = ''.join(text1)
                # mels, alignment_history, audios = self.do_synthesis(input_ids, self.tacotron2, self.mb_melgan, "TACOTRON", "MB-MELGAN")
                mels, audios = self.do_synthesis(input_ids, self.fastspeech2, self.mb_melgan, "FASTSPEECH2", "MB-MELGAN")
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
    # input_text = "신한생명과 오렌지라이프의 통합법인인 신한라이프는 1일 오전 서울 중구 본사에서 출범식을 개최했다고 밝혔다. 이날 행사에는 조용병 신한금융그룹회장과 그룹사 최고경영자, 성대규 신한라이프사장, 신입사원을 포함한 임직원 대표들이 참석했다."
    # input_text = "퓨렌스는 매년, 송년회, 우수사원 시상, 전사 권고연차를 실시하였습니다. 그러나, 2021년은 아래와 같은 상황을 고려하여, 부득이하게 송년회행사와 전사 권고연차를 진행하지 않기로 하였습니다. 우수사원 시상은 경영지원팀에서 개별적으로 찾아뵙고 진행예정입니다. 첫번째, 현재 진행중인 우리은행, 신한생명등에 프로젝트 참여인력은 연말 상관없이 근무예정. 두번째, 코비드 19 상황 고려. 열심히 근무하고 있는 임직원 분들의 노고에 감사드리며 경영지원팀에서도 지원을 아끼지 않겠습니다. 궁금하신 사항은 경영지원팀에 문의하여 주세요. 감사합니다."
    # input_text = "신한생명과 오렌지라이프의 통합법인인 신한라이프는, 1일 오전, 서울 중구 본사에서 출범식을 개최했다고 밝혔다. 이날 행사에는 조용병 신한금융그룹회장과, 그룹사 최고경영자, 성대규 신한라이프 사장, 신입사원을 포함한 임직원 대표들이 참석했다."
    # input_text = "세상을 살아가다 보면 수없이 많은 까다로운 문제에 부딪히게 됩니다. 그때마다 다 새롭게 해결책을 찾는 것은 너무 비효율적입니다. 미리 미리 삶의 안내자가 될 자신만의 방법과 해결 원칙들을 정리하고 숙지해 놓을 필요가 있습니다."
    # input_text = "정승기님, 안녕하세요? 매월 두 번째 월요일에 정승기님을 찾아오는 디지털 인사이트입니다. 행사 소식이 많습니다. 2월 24일에 두 건의 웨비나가 진행됩니다. 10시에는 구글 클라우드 테크 웨비나, 11시에는 실거래 자동검증 솔루션 퍼펙트윈 웨비나가 진행될 예정이에요. 사전등록하고 인사이트 얻어가세요! 오늘은 밸런타인데이라 외로운 편집장 춘이가 덜 외롭기 위한 이벤트를 하나 준비해 봤습니다. 많은 참여 부탁드려요!"
    input_text = "안녕하세요. LG CNS 입니다. 지난 2021년 12월 10일, 로그포제이 관련 보안상 치명적인 결점이 발견됨에 따라 긴급 조치가 필요한 동향을 공유 드립니다. 금번 로그포제이에서 발생한 보안 이슈는 대부분 인터넷 서비스에 영향이 있는 최상급 위험도를 가진 보안취약점으로, 웹서비스, 클라우드, IT 솔루션 등 광범위한 써비스에 사용되어 피해 발생 가능성이 높습니다. 악성코드 감염, 원격 명령 실행 등 심각한 피해를 방지하기 위한 보완 패치가 발표되었습니다. 취약점 정보와 함께 스캐너 등 공격툴도 공개되었으니 신속한 대응하시길 바랍니다. 아래, 상세 내용 보기 버튼을 누르시면 내용 및 대응방안을 확인하실 수 있습니다. 감사합니다."
    # input_text = "코로나 확진자가 폭발적으로 증가하고 있습니다. 현재 저희직원 3명이 자택격리 중입니다. 자택 격리중인 3명 모두 2차접종 후 90일 경과 되었습니다. 2차 접종후 90일이 경과되신 분은 빠른 시일내에 3차접종 하실 것을 권고드립니다. 본인과 가족의 건강을 위하여 백신접종 및 마스크 착용 등 개인방역을 철저히 준수하여 주시기 바랍니다."
    # input_text = "이 문장은 모델 학습에<pause>1</pause> 사용되지 않은 문장입니다."
    # input_text = "안녕하십니까. 영업팀 신서영 매니저입니다. 개인사유로 17일, 18일 연차를 사용합니다. 급한 용무가 있으실 경우, <digit>010-2681-4079</digit>로 연락 주십시오. 감사합니다. 신서영 드림."
    # input_text = "들의 콩깍지는 깐 콩깍지인가 안깐 콩깍지인가. 깐 콩깍지면 어떻고 안 깐 콩각지면 어떠냐. 깐 콩깍지나 안 깐 콩깍지나 콩깍지는 다 콩깍지인데. 상표 붙인 큰 깡통은 깐 깡통인가? 안 깐 깡통인가? 앞 집 팥죽은 붉은 팥 풋팥죽이고 , 뒷집 콩죽은 햇콩단콩 콩죽, 우리집 깨죽은 검은깨 깨죽인데 사람들은 햇콩 단콩 콩죽 깨죽 죽먹기를 싫어하더라."
    print(tts.text_to_voice(input_text))
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간