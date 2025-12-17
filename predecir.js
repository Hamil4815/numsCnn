
async function loadModel() {
  const response = await fetch("https://hamil4815.github.io/numsCnn/model.json");
  if (!response.ok) throw new Error("No se pudo cargar el modelo");
  return await response.json();
}
async function loadModel2() {
  const response = await fetch("https://hamil4815.github.io/numsCnn/model2.json");
  // const response = await fetch("http://192.168.56.1:5550/projectos/ActorCritic/xxx.json");
  if (!response.ok) throw new Error("No se pudo cargar el modelo2");
  return await response.json();
}
let model = null;
let model2 = null;
loadModel().then(m => {
  model = m;
  model = setear(model);
  console.log("Modelo cargado");
});
loadModel2().then(m2 => {
  model2 = m2;
  model2 = setear(model2);
  console.log("Modelo2 cargado");
});

// let model = []// solo asume que hay algo que ya uego lo pongo
// let model2 =[]// solo asume que hay algo que ya uego lo pongo

function maxPooling(image, poolSize = 2) {
  const imageHeight = image.length;
  const imageWidth = image[0].length;
  const outHeight = Math.ceil(imageHeight / poolSize);
  const outWidth = Math.ceil(imageWidth / poolSize);

  const mapa = Array.from({ length: outHeight }, () => Array(outWidth));

  for (let i = 0; i < imageHeight; i += poolSize) {
    const oi = i / poolSize | 0; // índice en salida
    for (let j = 0; j < imageWidth; j += poolSize) {
      const oj = j / poolSize | 0;

      let max = -Infinity;

      // recorremos bloque
      for (let di = 0; di < poolSize; di++) {
        const ni = i + di;
        if (ni >= imageHeight) break;

        const row = image[ni];
        for (let dj = 0; dj < poolSize; dj++) {
          const nj = j + dj;
          if (nj >= imageWidth) break;

          const val = row[nj];
          if (val > max) max = val;
        }
      }
      mapa[oi][oj] = max;
    }
  }

  return mapa;
}
function dimension(img) {
  return [img.length, img[0].length];
}
function normalizeSample(sample, mean = 12.2238, std = 53.7271) {
  // sample es una matriz 2D como [[1,2],[3,4]]
  const normInp = sample.map(row =>
    row.map(v => (v - mean) / std)
  );

  // Para que encaje con tu modelo (que espera [ [..] ] )
  return [normInp];
}
function isBatched4D(x) { // [N][C][H][W]
  return Array.isArray(x) && Array.isArray(x[0]) && Array.isArray(x[0][0]) && Array.isArray(x[0][0][0]);
}
function isBatched2D(x) { // [N][D]
  return Array.isArray(x) && Array.isArray(x[0]) && typeof x[0][0] === 'number';
}
function ensureBatch4D(x) {
  if (isBatched4D(x)) return x;
  return [x]; // wrap single sample
}
function ensureBatch2D(x) {
  if (isBatched2D(x)) return x;
  return [x];
}
class Conv2D {//mejora en la inicializacion de los pesos
  constructor(inChannels, outChannels, kernelSize) {
    this.tipo = "conv2d";
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.lastOutputShape = [this.outChannels, 3, 3];

    const K = kernelSize;

    // --- CAMBIO AQUÍ: Inicialización He (Kaiming) ---
    // Fan-in = canales de entrada * ancho * alto del kernel
    const fanIn = inChannels * K * K;
    const std = Math.sqrt(2 / fanIn);
    // ------------------------------------------------

    this.kernels = new Array(outChannels);
    for (let f = 0; f < outChannels; f++) {
      this.kernels[f] = new Array(inChannels);
      for (let c = 0; c < inChannels; c++) {
        const kc = new Array(K);
        for (let ky = 0; ky < K; ky++) {
          const row = new Float32Array(K);
          for (let kx = 0; kx < K; kx++) {
            // Usamos la desviación calculada
            row[kx] = (Math.random() - 0.5) * 2 * std;
          }
          kc[ky] = row;
        }
        this.kernels[f][c] = kc;
      }
    }
    // ... resto del código igual (bias, grad, etc)
    this.bias = new Float32Array(outChannels);
    this.gradBias = new Float32Array(outChannels);
    this.grad = new Array(outChannels);
    for (let f = 0; f < outChannels; f++) {
      this.grad[f] = new Array(inChannels);
      for (let c = 0; c < inChannels; c++) {
        const gkc = new Array(K);
        for (let ky = 0; ky < K; ky++) {
          gkc[ky] = new Float32Array(K);
        }
        this.grad[f][c] = gkc;
      }
    }
    this.input = null;
    this._tmpInputFlat = null;
    this._tmpDInputFlat = null;
  }
  // ... resto de métodos ...
  toJSON() {
    return {
      tipo: this.tipo,
      inChannels: this.inChannels,
      outChannels: this.outChannels,
      kernelSize: this.kernelSize,
      lastOutputShape: this.lastOutputShape,
      kernels: this.kernels,
      bias: this.bias,
    };
  }
  // Forward: empaqueta la entrada en un Float32Array por imagen y calcula convolución
  forward(batchInput) {
    const batch = ensureBatch4D(batchInput); // [N][C][H][W]
    // const batch = batchInput;
    // console.log("Conv2D forward batch shape:", tensorInfo(batch));
    this.input = batch;
    const N = batch.length;
    const C = batch[0].length;
    const H = batch[0][0].length;
    const W = batch[0][0][0].length;
    const K = this.kernelSize;
    const outH = H - K + 1;
    const outW = W - K + 1;

    // Output en la forma esperada por el resto del código
    const output = Array.from({ length: N }, () =>
      Array.from({ length: this.outChannels }, () =>
        Array.from({ length: outH }, () => Array(outW).fill(0))
      )
    );

    // preparar buffer plano de entrada (reusar si es posible)
    const inSize = C * H * W;
    if (!this._tmpInputFlat || this._tmpInputFlat.length < inSize) {
      this._tmpInputFlat = new Float32Array(inSize);
    }

    const kernels = this.kernels;
    const bias = this.bias;

    for (let n = 0; n < N; n++) {
      // pack input[n] -> _tmpInputFlat
      let p = 0;
      const img = batch[n];
      for (let c = 0; c < C; c++) {
        const plane = img[c];
        for (let y = 0; y < H; y++) {
          const row = plane[y];
          for (let x = 0; x < W; x++) {
            this._tmpInputFlat[p++] = row[x];
          }
        }
      }

      // compute conv using flat input but reading kernels via kernels[f][c][ky][kx]
      for (let f = 0; f < this.outChannels; f++) {
        const b = bias[f];
        for (let oy = 0; oy < outH; oy++) {
          for (let ox = 0; ox < outW; ox++) {
            let sum = 0.0;
            // inner product: ciclo por canales y kernel
            for (let c = 0; c < C; c++) {
              const inBase = c * (H * W);
              const kc = kernels[f][c];
              for (let ky = 0; ky < K; ky++) {
                const inRowStart = inBase + (oy + ky) * W + ox;
                const krow = kc[ky]; // Float32Array
                // sumar fila de kernel
                for (let kx = 0; kx < K; kx++) {
                  sum += this._tmpInputFlat[inRowStart + kx] * krow[kx];
                }
              }
            }
            output[n][f][oy][ox] = sum + b;
          }
        }
      }
    }

    return output;
  }

  // Backward: acumula gradientes en this.grad y this.gradBias; devuelve dInput en forma anidada
  backward(dOutBatch) {
    const batch = ensureBatch4D(dOutBatch); // [N][outChannels][outH][outW]
    const N = batch.length;
    const C = this.input[0].length;
    const H = this.input[0][0].length;
    const W = this.input[0][0][0].length;
    const K = this.kernelSize;
    const outH = batch[0][0].length;
    const outW = batch[0][0][0].length;

    const inSize = C * H * W;
    if (!this._tmpInputFlat || this._tmpInputFlat.length < inSize) {
      this._tmpInputFlat = new Float32Array(inSize);
    }
    if (!this._tmpDInputFlat || this._tmpDInputFlat.length < inSize) {
      this._tmpDInputFlat = new Float32Array(inSize);
    }

    const kernels = this.kernels;
    const grad = this.grad;
    // console.log( grad[0] );
    const gradBias = this.gradBias;

    // dInputs en la forma anidada esperada por el resto del código
    const dInputs = Array.from({ length: N }, () =>
      Array.from({ length: C }, () =>
        Array.from({ length: H }, () => Array(W).fill(0))
      )
    );

    for (let n = 0; n < N; n++) {
      // empacar input n a plano
      let p = 0;
      const img = this.input[n];
      for (let c = 0; c < C; c++) {
        const plane = img[c];
        for (let y = 0; y < H; y++) {
          const row = plane[y];
          for (let x = 0; x < W; x++) {
            this._tmpInputFlat[p++] = row[x];
          }
        }
      }

      // limpiar tmpDInputFlat
      this._tmpDInputFlat.fill(0);

      // recorrer deltas y acumular gradientes
      for (let f = 0; f < this.outChannels; f++) {
        const kcList = kernels[f];
        for (let oy = 0; oy < outH; oy++) {
          for (let ox = 0; ox < outW; ox++) {
            const delta = batch[n][f][oy][ox];
            gradBias[f] += delta;
            for (let c = 0; c < C; c++) {
              const inBase = c * (H * W);
              const gkc = grad[f][c];
              const kkc = kcList[c];
              for (let ky = 0; ky < K; ky++) {
                const inRowBase = inBase + (oy + ky) * W + ox;
                const krow = kkc[ky];
                const grow = gkc[ky];
                for (let kx = 0; kx < K; kx++) {
                  const inVal = this._tmpInputFlat[inRowBase + kx];
                  grow[kx] += inVal * delta; // acumular grad de kernel
                  this._tmpDInputFlat[inRowBase + kx] += krow[kx] * delta; // propagar a dInput plano
                }
              }
            }
          }
        }
      }

      // desempaquetar _tmpDInputFlat a dInputs[n]
      let q = 0;
      for (let c = 0; c < C; c++) {
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            dInputs[n][c][y][x] = this._tmpDInputFlat[q++];
          }
        }
      }
    }

    // console.log( dInputs[0] );
    return dInputs;
  }
}
class Linear {//mejora en la inicializacion de los pesos
  constructor(inSize, outSize) {
    this.tipo = "fc";
    this.inp = inSize;
    this.out = outSize;

    // --- CAMBIO AQUÍ: Inicialización He ---
    const std = Math.sqrt(2 / inSize);
    // --------------------------------------

    this.weight = Array.from({ length: outSize }, () =>
      Array.from({ length: inSize }, () => (Math.random() - 0.5) * 2 * std)
    );
    this.bias = Array(outSize).fill(0);

    // ... resto igual ...
    this.grad = Array.from({ length: outSize }, () => Array(inSize).fill(0));
    this.gradBias = Array(outSize).fill(0);
    this.lastInput = null;
  }
  // ... resto de métodos ...
  toJSON() {
    return {
      tipo: this.tipo,
      inp: this.inp,
      out: this.out,
      weight: this.weight,
      bias: this.bias,
    };
  }
  forward(x) {
    // console.log( 'antes',tensorInfo(x) );
    const batch = ensureBatch2D(x); // [N][inSize]
    // console.log( 'despues',tensorInfo(x) );
    this.lastInput = batch;
    const N = batch.length;
    const out = Array.from({ length: N }, () =>
      Array(this.weight.length).fill(0)
    ); // [N][outSize]
    for (let n = 0; n < N; n++) {
      for (let i = 0; i < this.weight.length; i++) {
        let sum = this.bias[i];
        for (let j = 0; j < this.weight[i].length; j++)
          sum += this.weight[i][j] * batch[n][j];
        out[n][i] = sum;
      }
    }
    return out;
  }

  backward(dout) {
    const dBatch = ensureBatch2D(dout); // [N][outSize]
    const N = dBatch.length;
    const inSize = this.weight[0].length;
    const outSize = this.weight.length;

    // NOTE: do NOT reset this.grad here. It must accumulate across batch processing.

    // dx: [N][inSize]
    const dx = Array.from({ length: N }, () => Array(inSize).fill(0));

    for (let n = 0; n < N; n++) {
      for (let i = 0; i < outSize; i++) {
        const g = dBatch[n][i];
        this.gradBias[i] += g;
        for (let j = 0; j < inSize; j++) {
          this.grad[i][j] += this.lastInput[n][j] * g;
          dx[n][j] += this.weight[i][j] * g;
        }
      }
    }

    return dx.length === 1 ? dx[0] : dx;
  }
}
class ReLU {
  constructor() { this.tipo = "relu"; this.mask = null }
  forward(x) {
    // x can be [N][D], [N][C][H][W], or single sample
    if (isBatched2D(x)) {
      const batch = x;
      this.mask = batch.map(row => row.map(v => v <= 0));
      return batch.map(row => row.map(v => Math.max(0, v)));
    }
    if (isBatched4D(x)) {
      const batch = x;
      this.mask = batch.map(sample => sample.map(ch => ch.map(row => row.map(v => v <= 0))));
      return batch.map(sample => sample.map(ch => ch.map(row => row.map(v => Math.max(0, v)))));
    }
    // single-case fallback
    if (typeof x[0] === 'number') {
      this.mask = x.map(v => v <= 0);
      return x.map(v => Math.max(0, v));
    }
    // conv single-sample
    this.mask = x.map(ch => ch.map(row => row.map(v => v <= 0)));
    return x.map(ch => ch.map(row => row.map(v => Math.max(0, v))));
  }
}
class MaxPool2D {
  constructor(size = 2) {
    this.tipo = "maxPool"
    this.size = size;
    this.indices = null
  }
  forward(batchInput) {
    const batch = ensureBatch4D(batchInput); // [N][C][H][W]
    const N = batch.length;
    const C = batch[0].length;
    const H = batch[0][0].length;
    const W = batch[0][0][0].length;
    const outH = Math.floor(H / this.size);
    const outW = Math.floor(W / this.size);

    const output = Array.from({ length: N }, () =>
      Array.from({ length: C }, () => Array.from({ length: outH }, () => Array(outW).fill(0)))
    );

    this.indices = Array.from({ length: N }, () =>
      Array.from({ length: C }, () => Array.from({ length: outH }, () => Array(outW))))

    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let i = 0; i < outH; i++) {
          for (let j = 0; j < outW; j++) {
            let maxVal = -Infinity;
            let maxIdx = [0, 0];
            for (let m = 0; m < this.size; m++) {
              for (let n2 = 0; n2 < this.size; n2++) {
                const val = batch[n][c][i * this.size + m][j * this.size + n2];
                if (val > maxVal) { maxVal = val; maxIdx = [i * this.size + m, j * this.size + n2]; }
              }
            }
            output[n][c][i][j] = maxVal;
            this.indices[n][c][i][j] = maxIdx;
          }
        }
      }
    }
    return output;
  }
}
class flat {
  constructor() {
    this.tipo = "flat"
    this.lastShape = null
  }
  forward(batchTensor3D) {
    const N = batchTensor3D.length
    const C = batchTensor3D[0].length
    const H = batchTensor3D[0][0].length
    const W = batchTensor3D[0][0][0].length

    this.lastShape = [C, H, W]  // <-- se guarda para el backward

    const OUT = Array.from({ length: N }, () => new Float32Array(C * H * W))
    for (let n = 0; n < N; n++) {
      let idx = 0
      for (let c = 0; c < C; c++) for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        OUT[n][idx++] = batchTensor3D[n][c][i][j]
      }
    }
    return OUT.map(a => Array.from(a))  // convertir a array normal
  }

}
function softmaxBatch(logitsBatch) {
  // logitsBatch: [N][C] -> returns [N][C]
  const batch = ensureBatch2D(logitsBatch);
  return batch.map(logits => {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(v => Math.exp(v - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  });
}
function setear(modelData) {
  let model = [];
  for (let i = 0; i < modelData.length; i++) {
    // console.log( model.length, i );
    // console.log( modelData[i]["tipo"] );
    if (modelData[i]["tipo"] == "relu") {
      model.push(new ReLU());
    }
    if (modelData[i]["tipo"] == "maxPool") {
      model.push(new MaxPool2D(modelData[i].size, modelData[i].size));
    }
    if (modelData[i]["tipo"] == "conv2d") {
      let conv = new Conv2D(
        modelData[i].inChannels,
        modelData[i].outChannels,
        modelData[i].kernelSize
      );
      conv.kernels = modelData[i].kernels;
      conv.bias = modelData[i].bias;
      model.push(conv);
    }
    if (modelData[i]["tipo"] == "fc") {
      let fc = new Linear(
        modelData[i].inp, modelData[i].out,
      );
      fc.weight = modelData[i].weight;
      fc.bias = modelData[i].bias;
      model.push(fc);
    }
    if (modelData[i]["tipo"] == "flat") {
      fc = modelData[i].weight;
      model.push(new flat());
    }
  }
  return new Sequential(model);
}
function evalBatchPreds(xBatch, model, returnProbs = false) {
  model.eval();
  const logits = model.forward(xBatch);          // [batchSize][numClasses]
  const probs = softmaxBatch(logits);           // misma forma
  const preds = probs.map(p => p.indexOf(Math.max(...p)));
  return { preds, logits: probs };
  // return { preds, logits };
}
class Sequential {
  constructor(...layers) {
    this.layers = layers.flat()
    this.training = true   // por defecto en modo training
  }
  train() {
    this.training = true
  }
  eval() {
    this.training = false
  }
  add(layer) {
    this.layers.push(layer)
  }
  forward(x) {
    let out = x
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i]
      if (typeof layer.forward !== 'function')
        throw new Error('Layer sin forward detectada')
      if (layer.forward.length >= 2) {
        out = layer.forward(out, this.training)
      } else {
        out = layer.forward(out)
      }
    }
    return out
  }
}
document.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById("canvas");
  // const ctx = canvas.getContext("2d");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const resultados = document.getElementById('resultados').getContext('2d');
  const borrar = document.getElementById("clearButton");
  const select = document.getElementById("miSelect");
  select.value= 'v2';
  let drawing = false;
  const miGrafica = new Chart(resultados, {
    type: 'bar', // Tipo de gráfica: 'line', 'pie', 'bar', etc.
    data: {
      labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
      datasets: [{
        label: ['m1','m2'],
        data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        backgroundColor: ['rgba(255, 99, 132, 0.2)', 'rgba(54, 162, 235, 0.2)', 'rgba(255, 206, 86, 0.2)', 'rgba(75, 192, 192, 0.2)'],
        borderColor: ['rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(255, 206, 86, 1)', 'rgba(75, 192, 192, 1)'],
        borderWidth: 1
      }],
      datasets: [
        { label: 'Modelo A', data: Array(10).fill(0),},
        { label: 'Modelo B', data: Array(10).fill(0) }
      ]
    },
    options: {
      plugins: {
        legend: { display: true },
        tooltip: {
          enabled: true,
          callbacks: {
            // label: ctx => ctx.raw.toFixed(3)
            label: ctx => `${(ctx.raw * 100).toFixed(1)} %`
          }
        }
      },
      animation: {
        duration: 2000,
        easing: 'easeOutQuart'
      },
      scales: {
        x: {
          beginAtZero: true,
          grid: { display: true },
        },
        y: {
          grid: { display: false },
          ticks: { display: false }
        }
      }
    }
  });

  function esPotenciaDe2(n) {
    return n > 0 && (n & (n - 1)) === 0;
  }
  function ajusta(img) {
    // la imagen stiene que ser mayor o igual a 32x32 y cuadrada pero tambien una potencia de 2
    let [x, y] = dimension(img);
    if (x > 32 && y == x && esPotenciaDe2(x)) {
      const factor = dimension(img)[0] / 32;
      // console.log(factor)
      return maxPooling(img, factor);
    }
  }
  function drawLine(x, y) {
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  }
  function canvasToMatrix() {
    const img = [];
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    for (let y = 0; y < canvas.height; y++) {
      const row = [];
      for (let x = 0; x < canvas.width; x++) {
        const index = (y * canvas.width + x) * 4;
        const alpha = data[index + 3];
        row.push(alpha > 0 ? alpha : 0);
      }
      img.push(row);
    }
    return img;
  }
  //-------------------------------------------
  canvas.addEventListener("mousedown", () => {
    drawing = true;
  });
  canvas.addEventListener("mouseup", () => {
    drawing = false;
    ctx.beginPath();
  });
  //-------------------------------------------
  canvas.addEventListener("mousemove", e => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    drawLine(e.clientX - rect.left, e.clientY - rect.top);
  });
  canvas.addEventListener("touchmove", e => {
    if (!drawing) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    drawLine(touch.clientX - rect.left, touch.clientY - rect.top);
  });
  //------------------------------------------
  borrar.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    resultados.clearRect(0, 0, canvas.width, canvas.height);
    miGrafica.data.datasets[0].data = Array(miGrafica.data.labels.length).fill(0);
    miGrafica.data.datasets[1].data = Array(miGrafica.data.labels.length).fill(0);
    miGrafica.update();
  });
  predecir.addEventListener("click", () => {
    let img = canvasToMatrix();
    img = ajusta(img);
    img = normalizeSample(img);
    let x1 = evalBatchPreds(img, model);
    let x2= evalBatchPreds(img, model2);
    console.log( x1.logits[0], x2.logits[0]);
    miGrafica.data.datasets[0].data = x1.logits[0];
    miGrafica.data.datasets[1].data = x2.logits[0];
    miGrafica.update();
  });
  select.addEventListener("change", (e) => {
    const valor = e.target.value;
    if (valor === "v1") {
      console.log("modo pequeño");
      canvas.width = 128;
      canvas.height = 128;
    }
    if (valor === "v2") {
      console.log("modo mediano");
      canvas.width = 256;
      canvas.height = 256;
    }
    if (valor === "v3") {
      console.log("modo grande");
      canvas.width = 512;
      canvas.height = 512;
    }
  });
});
