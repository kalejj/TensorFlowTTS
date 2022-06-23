import time
import os
import sys
import time
from datetime import datetime
import tensorflow as tf
import yaml
import numpy as np
import matplotlib.pyplot as plt
import re
import librosa
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor
import scipy.io.wavfile as wavf
import flask
from flask_restful import Resource, Api, reqparse
import io

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = flask.Flask(__name__)
api = Api(app)

# sunhi
# sunhi_tacotron2_config = AutoConfig.from_pretrained('examples/tacotron2/conf/tacotron2.kss.v1.yaml')
# sunhi_tacotron2 = TFAutoModel.from_pretrained(
#     config=sunhi_tacotron2_config,
#     pretrained_path="model_out/tacotron2/checkpoints/model-200000.h5",
#     name="tacotron2"
# )

sunhi_fastspeech2_config = AutoConfig.from_pretrained('examples/fastspeech2/conf/fastspeech2.kss.v1.yaml')
sunhi_fastspeech2 = TFAutoModel.from_pretrained(
    config=sunhi_fastspeech2_config,
    pretrained_path="model_out/fastspeech2/checkpoints/model-200000.h5",
    name="fastspeech2"
    )

sunhi_mb_melgan_config = AutoConfig.from_pretrained('examples/multiband_melgan/conf/multiband_melgan.v1.yaml')
sunhi_mb_melgan = TFAutoModel.from_pretrained(
    config=sunhi_mb_melgan_config,
    pretrained_path="model_out/mb-melgan/checkpoints/generator-1300000.h5",
    name="mb_melgan"
    )

# seoyoung
# young_tacotron2_config = AutoConfig.from_pretrained('examples/tacotron2/conf/tacotron2.kss.v1.yaml')
# young_tacotron2 = TFAutoModel.from_pretrained(
#     config=young_tacotron2_config,
#     pretrained_path="model_out/young_new_tacotron2/checkpoints/model-40000.h5",
#     name="tacotron2"
# )

young_fastspeech2_config = AutoConfig.from_pretrained('examples/fastspeech2/conf/fastspeech2.kss.v1.yaml')
young_fastspeech2 = TFAutoModel.from_pretrained(
    config=young_fastspeech2_config,
    pretrained_path="model_out/young_new_fastspeech2/checkpoints/model-4000.h5",
    name="fastspeech2"
    )

young_mb_melgan_config = AutoConfig.from_pretrained('examples/multiband_melgan/conf/multiband_melgan.v1.yaml')
young_mb_melgan = TFAutoModel.from_pretrained(
    config=young_mb_melgan_config,
    pretrained_path="model_out/young_mb-melgan/checkpoints/generator-1500000.h5",
    name="mb_melgan"
    )
processor = AutoProcessor.from_pretrained(pretrained_path="test/files/kss_mapper.json")

digit_dict = {'0': '공','1': '일','2': '이','3': '삼','4': '사','5': '오','6': '육','7': '칠','8': '팔','9': '구'}
email_dict = {"0": "공 ", "1": "일 ", "2": "이 ", "3": "삼 ", "4": "사 ", "5": "오 ", "6": "육 ", "7": "칠 ", "8": "팔 ", "9": "구 ", "A": "에이 ", "B": "비 ", "C": "씨 ", "D": "디 ", "E": "이 ", "F": "에프 ", "G": "쥐 ", "H": "에이치 ", "I": "아이 ", "J": "제이 ", "K": "케이 ", "L": "엘 ", "M": "엠 ", "N": "엔 ", "O": "오 ", "P": "피 ", "Q": "큐 ", "R": "알 ", "S": "에스 ", "T": "티 ", "U": "유 ", "V": "브이 ", "W": "더블유 ", "X": "엑스 ", "Y": "와이 ", "Z": "지 ", "@": "골뱅이 ", ".": "쩜 "}

class param(Resource):
    def get(self):
        return {'status': 'success'}

    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('text', type = str)
            parser.add_argument('speaker', type = str, required = False, default = 'sunhi')
            parser.add_argument('speed', type = float, required = False, default = 1.0)
            parser.add_argument('pitch', type = float, required = False, default = 1.0)
            args = parser.parse_args()
            _text = args['text']
            _speaker = args['speaker']
            _speed = args['speed']
            _pitch = args['pitch']
            start = time.time()
            _audio_path, _total_text = predict(_text, _speaker, _speed, _pitch)
            _time = time.time()-start

            return {
                'text': _text,
                'speaker': _speaker,
                'speed': _speed,
                'pitch': _pitch,
                'preprocessed_text': _total_text,
                'audio_path': _audio_path,
                'time': _time
                }
    
        except Exception as e:
            return {'error': str(e)}

api.add_resource(param, '/')
def predict(input_text, speaker, speed, pitch):
    # 입력 문장을 effect_tag, pause_tag, 문장([. ? !]) 단위로 split
    origin = re.split("\. |\? |! |<[ktml|pause]+>[^<>]*</[ktml|pause]+>", input_text)
    split = re.findall('\. |\? |! |<[ktml|pause]+>[^<>]*</[ktml|pause]+>', input_text)
    text_list = []
    for i in range(len(split)):
        text_list.append(origin[i].strip())
        text_list.append(split[i].strip())
    text_list.append(origin[-1])
    try:
        text_list.remove('')
    except:
        pass
    try:
        text_list.remove('.')
    except:
        pass
    try:
        text_list.remove('?')
    except:
        pass
    try:
        text_list.remove('!')
    except:
        pass

    # digit, email tag 한글로 변환
    for x in range(len(text_list)):
        # # digit_tag
        while '<digit>' in text_list[x]:
            start = text_list[x].find('<digit>')
            end = text_list[x].find('</digit>') + 8
            new = text_list[x][start:end].replace('<digit>', '').replace('</digit>', '')
            new = re.sub('[^0-9]', ' ', new)
            for i, j in digit_dict.items():
                new = new.replace(i, j)
            text_list[x] = text_list[x].replace(text_list[x][start:end], new)
            text_list[x] = text_list[x].replace('  ', ' ')
        # email_tag
        while '<email>' in text_list[x]:
            start = text_list[x].find('<email>')
            end = text_list[x].find('</email>') + 8
            new = text_list[x][start:end].replace('<email>', '').replace('</email>', '')
            new = new.upper()
            for i, j in email_dict.items():
                new = new.replace(i, j)
            text_list[x] = text_list[x].replace(text_list[x][start:end], ' ' + new)
            text_list[x] = text_list[x].replace('  ', ' ')
    print('input_text:', text_list)
    # TTS 변환
    total_audios = np.array([0], dtype = 'float32')
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
            if speaker == 'sunhi':
                if (text_list[i][-1] != '.') & (text_list[i][-1] != '?') & (text_list[i][-1] != '!'):
                    text_list[i] = text_list[i] + '.'
                input_ids, preprocessed_text = processor.text_to_sequence(text_list[i])
                text_list[i] = ''.join(preprocessed_text)
                mel_before, mel_outputs, duration_outputs, _, _ = sunhi_fastspeech2.inference(
                    tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
                    speed_ratios=tf.convert_to_tensor([float(speed)], dtype=tf.float32),
                    f0_ratios=tf.convert_to_tensor([float(pitch)], dtype=tf.float32),
                    energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                    )
                total_audio = sunhi_mb_melgan.inference(mel_outputs)[0, :, 0]
            elif speaker == 'young':
                if (text_list[i][-1] != '.') & (text_list[i][-1] != '?') & (text_list[i][-1] != '!'):
                    text_list[i] = text_list[i] + '.'
                input_ids, preprocessed_text = processor.text_to_sequence(text_list[i])
                text_list[i] = ''.join(preprocessed_text)
                mel_before, mel_outputs, duration_outputs, _, _ = young_fastspeech2.inference(
                    tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
                    speed_ratios=tf.convert_to_tensor([float(speed)], dtype=tf.float32),
                    f0_ratios=tf.convert_to_tensor([float(pitch)], dtype=tf.float32),
                    energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                    )
                total_audio = young_mb_melgan.inference(mel_outputs)[0, :, 0]
            total_audios = np.append(total_audios, total_audio, axis = 0)

        total_audios = np.append(total_audios, np.array([0]*7000, dtype = 'float32'))

    print('preprocessed_text:', text_list)
    
    cur_time = datetime.now()
    timestamp_str = cur_time.strftime("%Y%m%d_%H%M%S_%f")
    sample_rate = 22050
    output_audio = os.path.join('output/api',timestamp_str + '_' + speaker +'.wav')
    wavf.write(output_audio, sample_rate, total_audios)
    return output_audio, text_list

if __name__ == "__main__":
    app.run(port = '5001', debug = True)