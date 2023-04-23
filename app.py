import os
import tempfile
import psutil
from dotenv import load_dotenv
from flask import Flask, request
import openai
import pinecone
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tiktoken
from flask_cors import CORS

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

# Pinecone index name (change this to whatever you want)
index_name = 'jarvis'

# create the index in Pinecone if it doesn't already exist
if index_name not in pinecone.list_indexes():
  pinecone.create_index(
    index_name,
    dimension=1536,
    metric='cosine',
  )

# connect to index, define embedding model & input size, etc
index = pinecone.Index(index_name)
embed_model = "text-embedding-ada-002"
embed_length = 8191
embed_encode = 'cl100k_base'

# create Flask app
app = Flask(__name__)
CORS(app)


# RAM usage printer helps to monitor when/where use ramps up
def print_memory_usage():
  process = psutil.Process(os.getpid())
  mem_info = process.memory_info()
  print(f"Memory used: {mem_info.rss / (1024 * 1024)} MB")

# openai cookbook-- counting tokens!
def count_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Accepts up to 5 audio files. Sends to whisper. Embeds using ada. Uploads to Pinecone.
# this works really well for <20MB mp3 files (which are bigger when opened into an AudioSegment)
# but kinda sucks for bigger files, and breaks (with replit 4GB RAM?) around 100MB in my experience
# I'm sure there's better ways to handle processing audio files, or if someone built this into a
# product, you'd probably want to break up the audio before sending it here (like on device)
@app.route("/upload", methods=(["POST"]))
def upload():
  print("POST 'upload' running; loading file(s)")

  # was convenient during testing to set up Pinecone index again if not, since the server usually stays on
  pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT"))
  # create the index in Pinecone if it doesn't already exist
  if index_name not in pinecone.list_indexes():
    pinecone.create_index(
      index_name,
      dimension=1536,
      metric='cosine',
    )

  # initialize transcripts vector
  transcripts = []
  print("current pinecone index stats: ")
  print(index.describe_index_stats())

  # loop thru up to 5 audio files in POST form data
  for file_num in range(1, 6):
    file_key = f'audiofile{file_num}'
    if file_key not in request.files:
      print("file_key not found")
      break
    # print_memory_usage()
    # when testing, I used a lot of these to monitor RAM^

    file = request.files[file_key]
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
      temp_file.write(file.stream.read())
      temp_file.flush()

      # process the audio file-- remove its silence using a handy pydub function
      # this is normally the failure point for bigger files
      print("calling process_audio now")
      processed_audio = process_audio(temp_file.name)
      # convenient to look at the audio with silence removed to verify how well its tuned
      processed_audio.export(
        f'./output/silence_removed_audiofile{file_num}.mp3', format="mp3")

      # call split_audio_into_chunks to get 25MB chunks. Whisper maxes out at 25MB per call
      # i suspect there are a lot of remaining issues to be worked out here-- this probably
      # breaks in a lot of edge cases
      print("Got the processed audio, now going to split into 25MB chunks")
      chunks = split_audio_into_chunks(processed_audio)
      print("got chunks! length of chunks vector: ")
      print(len(chunks))

      # loop thru the <25MB chunks and transcribe them with whisper
      for chunk in chunks:
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix='.mp3') as chunk_temp_file:
          chunk.export(chunk_temp_file.name, format="mp3")
          chunk_temp_file.flush()

          # split_audio_into_chunks often returns a chunk that's too small to be sent to whisper,
          # even with high silence threshold, not sure why, only spent a few min trying to debug and moved on
          duration_ms = len(chunk)
          if duration_ms > 500:
            with open(chunk_temp_file.name, "rb") as audio_file:
              print("calling whisper now!")
              transcript = openai.Audio.transcribe("whisper-1", audio_file)
              transcripts.append(transcript["text"])

              # printing the transcripts during testing is helpful!
              # print("(looping thru chunks) current transcripts vector: ")
              # print(transcripts)
          else:
            print("empty chunk here, skipping!")

  print("done with for loop, creating embeddings now")
  # this creates an embedding for each ~25MB chunk file. It seems clear you could tune
  # the embedding "size" a lot and play with the top_k nearest neighbors to tune how it
  # responds. For example if you wanted a lot of different context across the past week
  # you could tune to have many little embeddings, or if you want it to grasp ~a conversation
  # less but bigger embeddings might be best. I'm making this up so this could be a bad theory
  embeddings = [
    openai.Embedding.create(input=[t],
                            engine=embed_model)['data'][0]['embedding']
    for t in transcripts
  ]

  # found it helpful for testing to reference all the data that Pinecone had
  print("embeddings made, saving transcripts to an output text file!")
  with open("transcripts_output.txt", "w") as file:
    for transcript in transcripts:
      file.write(transcript + "\n" + "-------" + "\n")

  print("done, preparing to upsert to Pinecone")
  # to avoid overwritting in pinecone, grab the current vector count and create IDs from there
  index_stats = index.describe_index_stats()
  vector_count = index_stats['total_vector_count']
  ids = [str(i + vector_count) for i in range(len(transcripts))]

  # upload the new embeddings in the transcripts vector to pinecone
  to_upsert = list(
    zip(ids, embeddings, [{
      'transcript': t
    } for t in transcripts]))
  index.upsert(vectors=to_upsert)
  print("done uploading to pinecone!")
  return 'Audio files processed and transcripts uploaded to Pinecone!'


# POST request takes in query, calls 'retrieve' to generate prompt, then sends prompt into GPT 3.5 and
# returns a response
@app.route("/question", methods=(["POST"]))
def question():
  print("POST 'question' running.")

  # grab query into string
  query = request.form['query']
  # grab k-neighbors to query
  query_with_contexts = retrieve(query)
  print("full query: ")
  print(query_with_contexts)

  # send full prompt to gpt
  answer = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                        messages=[{
                                          "role":
                                          "system",
                                          "content":
                                          "you're a helpful assistant."
                                        }, {
                                          "role":
                                          "user",
                                          "content":
                                          query_with_contexts
                                        }])

  answer_str = answer['choices'][0]['message']['content']
  # return answer!
  return answer_str


# intake an audio file, remove its silence and return.
def process_audio(file):
  print("Getting an AudioSegment from the file... (inside process_audio)")
  audio = AudioSegment.from_file(file)

  # get the size of the input audio
  audio_bytes = audio.get_array_of_samples().tobytes()
  audio_size = len(audio_bytes)
  print("audio_size in bytes before removing silence is: ")
  print(audio_size)
  print("length of audio in minutes before removing silence is: ")
  audio_duration_minutes = audio.duration_seconds / 60
  print(audio_duration_minutes)

  print("got audioSegment, about to call split_on_silence")

  # handy pydub function that removes silence from an audio recording under the silence_thresh value (in dBFS)
  # I found -35 was really good for my recorder, but this could probably be tuned quite a bit to produce better results
  # This function seemed to be the bottleneck with trying to process bigger files!
  # NOTE: THIS CAN TAKE A WHILE
  chunks = split_on_silence(audio,
                            min_silence_len=1000,
                            silence_thresh=-35,
                            keep_silence=200)
  print("ok removed silence, recombining chunks now")

  # NOTE: THIS ALSO TAKES A WHILE
  combined = AudioSegment.empty()
  for chunk in chunks:
    combined += chunk

  print("returning combined chunks")
  # get the size of the output audio
  audio_bytes = combined.get_array_of_samples().tobytes()
  audio_size = len(audio_bytes)
  print("audio_size in bytes after removing silence is: ")
  print(audio_size)
  print("length of audio in minutes after removing silence is: ")
  audio_duration_minutes = combined.duration_seconds / 60
  print(audio_duration_minutes)
  return combined


# split audio into <25MB chunks. input is an AudioSegment and returns a list of AudioSegments.
# note that the audiosegment size is greater than the mp3 input file size
# i dont think this fully works as intended-- often returns chunks smaller than 0.1s or other strange things
# didn't spend much time debugging this!
def split_audio_into_chunks(audio, max_size_bytes=24 * 1024 * 1024):

  # get the size of the input audio
  audio_bytes = audio.get_array_of_samples().tobytes()
  audio_size = len(audio_bytes)

  # just return if no need to chunk!
  if audio_size <= max_size_bytes:
    print("audio size is smaller than 25MB, so just returning it!")
    return [audio]

  # calculate the number of chunks to break the audio into
  num_chunks = int(
    (audio.frame_count() * audio.frame_width + max_size_bytes - 1) //
    max_size_bytes)

  print("num chunks should be: ")
  print(num_chunks)

  # compute the length of each chunk if dividing evenly, then gets remainder
  base_chunk_length_ms = round((audio.duration_seconds * 1000) / num_chunks)
  remainder = round(audio.duration_seconds * 1000) % num_chunks
  print("chunk length in ms should be: ")
  print(base_chunk_length_ms)

  print("audio input length in ms is: ")
  print(audio.duration_seconds * 1000)

  # loop thru chunks and break them ~evenly by distributing the remainder
  chunks = []
  start = 0
  for i in range(num_chunks):
    chunk_length_ms = base_chunk_length_ms + (1 if i < remainder else 0)
    end = start + chunk_length_ms
    chunks.append(audio[int(start):int(end)])
    start = end

  print("returning <25MB chunks!")
  return chunks


# takes in query, creates embedding using Ada, sends to Pinecone to grab k-nearest-neighbors
# & returns a prompt with context
def retrieve(query):
  # get the embedding from Ada
  res = openai.Embedding.create(input=[query], engine=embed_model)
  xq = res['data'][0]['embedding']

  # get relevant contexts from Pinecone
  # this usually works pretty well for specific queries
  res = index.query(xq, top_k=5, include_metadata=True)
  contexts = [x['metadata']['transcript'] for x in res['matches']]
  # print("contexts: ")
  # print(contexts)

  # build prompt with the retrieved contexts included
  prompt_start = ("You are a helpful assistant that has data collected from an audio recording of my conversations. Answer the question based on the context below.\n\n" +
                  "Context:\n")
  prompt_end = (f"\n\nQuestion: {query}\nAnswer:")
  prompt = "placeholder prompt"

  # append contexts until hitting gpt 3.5 input token limit (4096, but using 3800 to allow for response)
  # could obviously fine-tune this much more-- single embedding can be ~1k tokens, so you could fit a lot more
  # if you instead just cut words off or something, or sized your embeddings just right!
  for i in range(1, len(contexts) + 1):
    joined_contexts = "\n\n---\n\n".join(contexts[:i])
    prompt = (prompt_start + joined_contexts + prompt_end)
    if count_tokens(prompt, embed_encode) >= 3800:
      prompt = (prompt_start + "\n\n---\n\n".join(contexts[:i - 1]) + prompt_end)
      break

  return prompt


# runs the app!
app.run(host='0.0.0.0', port=8080)
