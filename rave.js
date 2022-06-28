let audioCtx = null;
let inputBuffer = null;
let outputBuffer = null;
let isRecording = false;
let stream = null;
let recorder = null;
let chunks = [];

// ENABLING AUDIO CONTEXT
const enableAudioCtx = () => {
  if (audioCtx != null) return;
  console.log("enabling audio");
  audioCtx = new (AudioContext || webkitAudioContext)();
};

// UPLOAD VARIOUS SOURCES TO MEMORY
const toogleRecording = async () => {
  console.log("toogle recording");
  let recordButton = document.getElementById("record-button");
  if (!isRecording) {
    recordButton.value = "Stop recording";
    isRecording = true;
    chunks = [];
    try {
      // GET INPUT STREAM AND CREATE RECORDER
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recorder = new MediaRecorder(stream);

      // ON STOP FUNCTION
      recorder.onstop = function (e) {
        let blob = new Blob(chunks, { type: "audio/ogg; codecs=opus" });
        chunks = [];
        let audioURL = URL.createObjectURL(blob);

        urlToBuffer(audioURL).then((buffer) => {
          inputBuffer = buffer;
          let playButton = document.getElementById("play_input");
          let ravifyButton = document.getElementById("ravify_button");
          playButton.disabled = false;
          ravifyButton.disabled = false;
        });
      };

      recorder.ondataavailable = function (e) {
        chunks.push(e.data);
      };

      recorder.start();
      /* use the stream */
    } catch (err) {
      console.log(err);
    }
  } else {
    recordButton.value = "Record from microphone";
    isRecording = false;
    recorder.stop();
    stream.getTracks().forEach((track) => track.stop());
  }
};

const loadUploadedFile = () => {
  enableAudioCtx();

  let fileInput = document.getElementById("audio-file");
  let ravifyButton = document.getElementById("ravify_button");
  if (fileInput.files[0] == null) return;

  let playButton = document.getElementById("play_input");
  playButton.disabled = true;
  var reader1 = new FileReader();
  reader1.onload = function (ev) {
    audioCtx.decodeAudioData(ev.target.result).then(function (buffer) {
      buffer = tensorToBuffer(bufferToTensor(buffer));
      inputBuffer = buffer;
      playButton.disabled = false;
      ravifyButton.disabled = false;
    });
  };
  reader1.readAsArrayBuffer(fileInput.files[0]);
};

const loadCantina = async () => {
  let playButton = document.getElementById("play_input");
  let ravifyButton = document.getElementById("ravify_button");
  playButton.disabled = true;

  let buffer = await urlToBuffer("/ravejs/default.mp3");
  buffer = tensorToBuffer(bufferToTensor(buffer));
  inputBuffer = buffer;
  playButton.disabled = false;
  ravifyButton.disabled = false;
};

const urlToBuffer = async (url) => {
  enableAudioCtx();
  const audioBuffer = await fetch(url)
    .then((res) => res.arrayBuffer())
    .then((ArrayBuffer) => audioCtx.decodeAudioData(ArrayBuffer));
  return audioBuffer;
};

// PLAY BUFFER
const playBuffer = (buffer) => {
  enableAudioCtx();
  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);
  source.start();
};

const playInput = () => {
  if (inputBuffer == null) return;
  playBuffer(inputBuffer);
};

const playOutput = () => {
  if (outputBuffer == null) return;
  playBuffer(outputBuffer);
};

// PROCESSING
const transfer = async () => {
  if (inputBuffer == null) return;
  console.log("transfer in progress...");
  outputBuffer = await raveForward(inputBuffer);
  make_download(outputBuffer, outputBuffer.getChannelData(0).length);
};

const bufferToTensor = (buffer) => {
  let b = buffer.getChannelData(0);
  let cropped_data = [];
  let cropped_length = Math.min(10 * audioCtx.sampleRate, b.length);
  for (let i = 0; i < cropped_length; i++) {
    cropped_data.push(b[i]);
  }
  const inputTensor = new ort.Tensor("float32", cropped_data, [
    1,
    1,
    cropped_length,
  ]);
  return inputTensor;
};

const tensorToBuffer = (tensor) => {
  let len = tensor.dims[2];
  let buffer = audioCtx.createBuffer(1, len, audioCtx.sampleRate);
  channel = buffer.getChannelData(0);
  for (let i = 0; i < buffer.length; i++) {
    channel[i] = isNaN(tensor.data[i]) ? 0 : tensor.data[i];
  }
  return buffer;
};

const raveForward = async (buffer) => {
  let model_name = document.getElementById("model");
  let playButton = document.getElementById("play_output");
  let ravifyButton = document.getElementById("ravify_button");
  ravifyButton.disabled = true;
  playButton.disabled = true;
  let inputTensor = bufferToTensor(buffer);
  let session = await ort.InferenceSession.create(model_name.value);
  let feeds = { audio_in: inputTensor };
  let audio_out = (await session.run(feeds)).audio_out;
  audio_out = tensorToBuffer(audio_out);
  playButton.disabled = false;
  ravifyButton.disabled = false;
  return audio_out;
};
