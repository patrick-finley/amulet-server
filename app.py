import os
from io import BufferedReader
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/whats_my_val", methods=(["GET"]))  # create a route for the app
def whats_my_val():  # create a function that will be called when the route is requested
  print('oh?')
  response = f'here you go friend, your val is ' + request.headers['secret_value']
  print('ah, someone just got something')
  return response


@app.route("/get_transcript", methods=(["POST"]))  # create a route for the app
def get_transcript():  # create a function that will be called when the route is requested
  secret_value = request.headers['secret_value']
  print(secret_value)

  print(f'ah, you were sent a secret value: {secret_value}')
  if secret_value == os.getenv('SECRET_VAL'):
    print('loading file')
    try:
        file = request.files['audiofile']
        
    except Exception as e:
        print('error loading file')
        print(e)
        exit()

    print('ok, about to call whisper!')
    
    # this changes the FileStorage type to a BufferedReader type, which I guess OpenAI needs
    file.name = file.filename
    file = BufferedReader(file)

    transcript = openai.Audio.transcribe("whisper-1", file)

    print('whisper is done!')
    transcript_str = transcript["text"]  # accessing the "text" from the JSON file that OpenAI returns

    return 'thanks pal, here is your transcript: ' + transcript_str
  else:
    return 'your key & value were incorrect, sorry!'


@app.route("/get_summary", methods=(["POST"]))
def get_summary():
   secret_value = request.headers['secret_value']
   print(f'ah, you were sent a secret value: {secret_value}')

   if secret_value == os.getenv('SECRET_VAL'):

    print('loading file')
    try:
        file = request.files['audiofile']
    except Exception as e:
        print('error loading file')
        print(e)
        exit()
    file = request.files['audiofile']

    print('ok, about to call whisper!')

    # this changes the FileStorage type to a BufferedReader type, which I guess OpenAI needs
    file.name = file.filename
    file = BufferedReader(file)

    transcript = openai.Audio.transcribe("whisper-1", file)

    print('whisper is done!')
    transcript_str = transcript["text"]  # accessing the "text" from the JSON file that OpenAI returns

    summary_request = request.form['summary_request']
    prompt = request.form['prompt']
    print('ok got the transcript & summary request & prompt, now sending it to gpt3.5: ' + summary_request + ' prompt: ' + prompt)
    
    summary = openai.ChatCompletion.create(
       model="gpt-3.5-turbo",
       messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript_str},
            {"role": "user", "content": summary_request}
       ]
    )

    summary_str = summary['choices'][0]['message']['content']

    print(summary_str)
    return 'thanks pal, here is your summary: ' + summary_str
   else:
    return 'your secret value was incorrect, sorry!'


app.run(host='0.0.0.0', port=8080)


######## Anant's code ########

# import os
# import openai
# from dotenv import load_dotenv
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# def get_transcript(audio_file):
#     response = openai.Audio.create(
#         file=audio_file.stream,
#         purpose="transcription",
#     )

#     transcript = response.get("text")
#     return transcript

# def get_summary(transcript):
#     prompt = f"Please provide a summary of the following transcript:\n{transcript}"
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=prompt,
#         max_tokens=100,
#         n=1,
#         stop=None,
#         temperature=0.5,
#     )

#     summary = response.choices[0].text.strip()
#     return summary

# @app.route('/api/summarize', methods=['POST'])
# def summarize_audio():
#     audio_file = request.files['audio']
#     transcript = get_transcript(audio_file)
#     summary = get_summary(transcript)
#     return jsonify(summary=summary)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
