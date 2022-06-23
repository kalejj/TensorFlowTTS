import time
import os
import sys
import time
from datetime import datetime
import tensorflow as tf
import time
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
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse
import io
from types import MethodType
from flask import Flask, render_template, request, redirect, url_for
from flask.globals import session
import requests
import urllib

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# sunhi
sunhi_fastspeech2_config = AutoConfig.from_pretrained('examples/fastspeech2/conf/fastspeech2.kss.v1.yaml')
sunhi_fastspeech2 = TFAutoModel.from_pretrained(
    config=sunhi_fastspeech2_config,
    pretrained_path="tacotron2-200k.h5",
    name="fastspeech2"
    )

sunhi_mb_melgan_config = AutoConfig.from_pretrained('examples/multiband_melgan/conf/multiband_melgan.v1.yaml')
sunhi_mb_melgan = TFAutoModel.from_pretrained(
    config=sunhi_mb_melgan_config,
    pretrained_path="model_out/mb-melgan/checkpoints/generator-1300000.h5",
    name="mb_melgan"
    )


processor = AutoProcessor.from_pretrained(pretrained_path="test/files/kss_mapper.json")

digit_dict = {'0': '공','1': '일','2': '이','3': '삼','4': '사','5': '오','6': '육','7': '칠','8': '팔','9': '구'}
email_dict = {"0": "공 ", "1": "일 ", "2": "이 ", "3": "삼 ", "4": "사 ", "5": "오 ", "6": "육 ", "7": "칠 ", "8": "팔 ", "9": "구 ", "A": "에이 ", "B": "비 ", "C": "씨 ", "D": "디 ", "E": "이 ", "F": "에프 ", "G": "쥐 ", "H": "에이치 ", "I": "아이 ", "J": "제이 ", "K": "케이 ", "L": "엘 ", "M": "엠 ", "N": "엔 ", "O": "오 ", "P": "피 ", "Q": "큐 ", "R": "알 ", "S": "에스 ", "T": "티 ", "U": "유 ", "V": "브이 ", "W": "더블유 ", "X": "엑스 ", "Y": "와이 ", "Z": "지 ", "@": "골뱅이 ", ".": "쩜 "}

def replace_pitch(x):
    if x == '1.0':
        a = '0'
    elif x[0] == '1':
        a = '+' + str(int(x[-1]) // 2)
    else:
        a = '-' + str(10 - int(x[-1]))
    return a

def replace_speed(x):
    if x == '1.0':
        a = '0'
    elif x[0] == '1':
        a = '-' + x[-1]
    else:
        a = '+' + str(10 - int(x[-1]))
    return a

app = Flask(__name__)
 
@app.route("/", methods = ['GET', 'POST'])
def search():
    if request.method == 'GET':
        return render_template('sunhi_search.html')
    elif request.method == 'POST':
        text = request.form['need_decoding_str']
        speed = request.form['spd']
        pitch = request.form['pitch']
        speaker = request.form['speaker']
        replaced_speed = replace_speed(speed)
        replaced_pitch = replace_pitch(pitch)
        text = text.replace('\r\n', ' ')
        start = time.time()
        audio_path, prepro_text = predict(text, speed, pitch, speaker)
        _time = time.time() - start
        if speaker == 'young':
            speaker = 'seoyoung'
        return render_template('sunhi_search.html', arg = audio_path, text = text, spd = replaced_speed, pitch = replaced_pitch, speaker = speaker, text_list = prepro_text, time = _time)
def predict(input_text, speed = 1.0, pitch = 1.0, speaker = 'sunhi'):
    start = time.time()
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
            elif speaker == 'japanese':
                input_ids = jp_processor.text_to_sequence(text_list[i], inference = True)
                mel_before, mel_outputs, duration_outputs, _, _ = japan_fastspeech2.inference(
                    tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
                    speed_ratios=tf.convert_to_tensor([float(speed)], dtype=tf.float32),
                    f0_ratios=tf.convert_to_tensor([float(pitch)], dtype=tf.float32),
                    energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                    )
                total_audio = japan_mb_melgan.inference(mel_outputs)[0, :, 0]
            total_audios = np.append(total_audios, total_audio, axis = 0)

        total_audios = np.append(total_audios, np.array([0]*7000, dtype = 'float32'))

    print('3:', text_list)
    
    cur_time = datetime.now()
    timestamp_str = cur_time.strftime("%Y%m%d_%H%M%S_%f")
    sample_rate = 22050
    output_audio = os.path.join('static',timestamp_str + '_' + speaker + '.wav')
    wavf.write(output_audio, sample_rate, total_audios)
    _time = time.time()-start

    return output_audio, text_list

    


if __name__ == "__main__":
    app.run(host='0.0.0.0', port = '5000')