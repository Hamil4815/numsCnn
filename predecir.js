

async function loadModel() {
  const response = await fetch("model.json"); // ruta relativa al HTML
  if (!response.ok) throw new Error("No se pudo cargar el modelo");
  return await response.json();
}
// uso
let model = null;
loadModel().then(m => {
  model = m;
  model = setear(model);
  console.log("Modelo cargado");
});

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

//-------------------------------------------------------
function normalizeSample(sample, mean=12.223856689453125, std=53.727097888564764) {
  // sample es una matriz 2D como [[1,2],[3,4]]
  const normInp = sample.map(row =>
    row.map(v => (v - mean) / std)
  );

  // Para que encaje con tu modelo (que espera [ [..] ] )
  return [normInp];
}
// -------------------- Utilidades para convolución --------------------
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
// -------------------- Layers --------------------
class Linear {
  constructor(inSize, outSize) {
    this.tipo= "fc";
    this.inp= inSize;
    this.out= outSize;
    this.weight = Array.from({ length: outSize }, () =>
      Array.from({ length: inSize }, () => Math.random() * 0.01)
    );
    this.bias = Array(outSize).fill(0);
    this.lastInput = null;
  }
  forward(x) {
    const batch = ensureBatch2D(x); // [N][inSize]
    this.lastInput = batch;
    const N = batch.length;
    const out = Array.from({length: N}, () => Array(this.weight.length).fill(0)); // [N][outSize]
    for (let n = 0; n < N; n++) {
      for (let i = 0; i < this.weight.length; i++) {
        let sum = this.bias[i];
        for (let j = 0; j < this.weight[i].length; j++) sum += this.weight[i][j] * batch[n][j];
        out[n][i] = sum;
      }
    }
    return out.length === 1 ? out[0] : out;
  }

}
class Conv2D {
  // Conv2D compatible con SGD (kernels y grad en forma anidada),
  // pero con optimizaciones internas (Float32Array rows, buffers planos, reuso).
  constructor( inChannels, outChannels, kernelSize ) {
    this.tipo = "conv2d"
    this.inChannels = inChannels
    this.outChannels = outChannels
    this.kernelSize = kernelSize
    this.lastOutputShape = [ this.outChannels, 3, 3 ]

    const K = kernelSize
    const K2 = K * K
    this.kernels = new Array( outChannels )
    for (let f = 0; f < outChannels; f++) {
      this.kernels[f] = new Array( inChannels )
      for (let c = 0; c < inChannels; c++) {
        // cada kernel es una matriz K x K; la representamos como array de filas Float32Array
        const kc = new Array( K )
        for (let ky = 0; ky < K; ky++) {
          const row = new Float32Array( K )
          for (let kx = 0; kx < K; kx++) row[kx] = (Math.random() - 0.5) * 0.1
          kc[ky] = row
        }
        this.kernels[f][c] = kc
      }
    }
    this.bias = new Float32Array( outChannels )
    this.input = null

  }
  forward( batchInput ) {
    const batch = ensureBatch4D( batchInput ) // mantiene tu convención [N][C][H][W]
    this.input = batch
    const N = batch.length
    const C = batch[0].length
    const H = batch[0][0].length
    const W = batch[0][0][0].length
    const K = this.kernelSize
    const outH = H - K + 1
    const outW = W - K + 1

    // Output en la forma esperada por el resto del código
    const output = Array.from({ length: N }, () =>
      Array.from({ length: this.outChannels }, () =>
        Array.from({ length: outH }, () => Array( outW ).fill(0))
      )
    )

    // preparar buffer plano de entrada (reusar si es posible)
    const inSize = C * H * W
    if (!this._tmpInputFlat || this._tmpInputFlat.length < inSize) {
      this._tmpInputFlat = new Float32Array( inSize )
    }

    const kernels = this.kernels
    const bias = this.bias

    for (let n = 0; n < N; n++) {
      // pack input[n] -> _tmpInputFlat
      let p = 0
      const img = batch[n]
      for (let c = 0; c < C; c++) {
        const plane = img[c]
        for (let y = 0; y < H; y++) {
          const row = plane[y]
          for (let x = 0; x < W; x++) {
            this._tmpInputFlat[p++] = row[x]
          }
        }
      }

      // compute conv using flat input but reading kernels via kernels[f][c][ky][kx]
      for (let f = 0; f < this.outChannels; f++) {
        const b = bias[f]
        for (let oy = 0; oy < outH; oy++) {
          for (let ox = 0; ox < outW; ox++) {
            let sum = 0.0
            // inner product: ciclo por canales y kernel
            for (let c = 0; c < C; c++) {
              const inBase = c * (H * W)
              const kc = kernels[f][c]
              for (let ky = 0; ky < K; ky++) {
                const inRowStart = inBase + (oy + ky) * W + ox
                const krow = kc[ky] // Float32Array
                // sumar fila de kernel
                for (let kx = 0; kx < K; kx++) {
                  sum += this._tmpInputFlat[inRowStart + kx] * krow[kx]
                }
              }
            }
            output[n][f][oy][ox] = sum + b
          }
        }
      }
    }

    return output
  }
}
// -------------------- Activations / Pooling / Dropout --------------------
class ReLU {
  constructor() { this.tipo= "relu"; this.mask = null  }
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
    this.tipo= "maxPool"
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

    const output = Array.from({length: N}, () =>
      Array.from({length: C}, () => Array.from({length: outH}, () => Array(outW).fill(0)))
    );

    this.indices = Array.from({length: N}, () =>
      Array.from({length: C}, () => Array.from({length: outH}, () => Array(outW))))

    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let i = 0; i < outH; i++) {
          for (let j = 0; j < outW; j++) {
            let maxVal = -Infinity;
            let maxIdx = [0,0];
            for (let m = 0; m < this.size; m++) {
              for (let n2 = 0; n2 < this.size; n2++) {
                const val = batch[n][c][i*this.size + m][j*this.size + n2];
                if (val > maxVal) { maxVal = val; maxIdx = [i*this.size + m, j*this.size + n2]; }
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
class Dropout {
  constructor(p) {
    this.tipo= "drop";
    this.p = p;
    this.mask = null; 
  }
  forward(x, training) {
    const batch = ensureBatch2D(x);
    if (!training) {
      // en eval devolvemos input y creamos máscara de 1s
      this.mask = batch.map(row => row.map(() => 1));
      return batch.length === 1 ? batch[0] : batch;  // <-- aquí, nada de outOrSingle
    }
    this.mask = batch.map(row => row.map(() => Math.random() > this.p ? 1 : 0));
    const out = batch.map((row, n) => row.map((v, i) => v * this.mask[n][i]));
    return out.length === 1 ? out[0] : out;
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

    const OUT = Array.from({length: N}, () => new Float32Array(C*H*W))
    for (let n = 0; n < N; n++) {
      let idx = 0
      for (let c = 0; c < C; c++) for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        OUT[n][idx++] = batchTensor3D[n][c][i][j]
      }
    }
    return OUT.map(a => Array.from(a))  // convertir a array normal
  }

}
// -------------------- Softmax --------------------
function softmaxBatch(logitsBatch) {
  // logitsBatch: [N][C] -> returns [N][C]
  const batch = ensureBatch2D(logitsBatch);
  return batch.map(logits => {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(v => Math.exp(v - maxLogit));
    const sum = exps.reduce((a,b) => a+b, 0);
    return exps.map(e => e / sum);
  });
}
//----------------------Setear modelo----------------------
function setear(modelData) {
  let model= [];
  for (let i = 0; i < modelData.length; i++) {
    // console.log( model.length, i );
    // console.log( modelData[i]["tipo"] );
    if (modelData[i]["tipo"]=="relu") {
      model.push(new ReLU());
    }
    if (modelData[i]["tipo"]=="maxPool") {
      model.push(new MaxPool2D(modelData[i].size,modelData[i].size));
    }
    if (modelData[i]["tipo"]=="conv2d") {
      let conv= new Conv2D(
        modelData[i].inChannels,
        modelData[i].outChannels,
        modelData[i].kernelSize
      );
      conv.kernels= modelData[i].kernels;
      conv.bias= modelData[i].bias;
      model.push(conv);
    }
    if (modelData[i]["tipo"]=="fc") {
      let fc= new Linear(
        modelData[i].inp,modelData[i].out,
      );
      fc.weight= modelData[i].weight;
      fc.bias= modelData[i].bias;
      model.push(fc);
    }
    if (modelData[i]["tipo"]=="flat") {
      fc= modelData[i].weight;
      model.push(new flat());
    }
    if (modelData[i]["tipo"]=="drop") {
      model.push(new Dropout(modelData[i].p));
    }
  }
  return new Sequential(model);
}
function evalBatchPreds(xBatch, model, returnProbs = false) {
  model.eval();
  const logits = model.forward(xBatch);          // [batchSize][numClasses]
  const probs = softmaxBatch(logits);           // misma forma
  const preds = probs.map(p => p.indexOf(Math.max(...p)));
  return preds;
}
class Sequential {
  constructor ( ...layers ) {
    this.layers = layers.flat()
    this.training = true   // por defecto en modo training
  }
  train() { 
    this.training = true 
  }
  eval() { 
    this.training = false 
  }
  add ( layer ) {
    this.layers.push(layer)
  }
  forward ( x ) {
    let out = x
    for ( let i = 0; i < this.layers.length; i++ ) {
      const layer = this.layers[i]
      if ( typeof layer.forward !== 'function' ) 
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
// const raw = document.getElementById("model-data").textContent;
// const model = JSON.parse(raw);
document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById("canvas");
    // const ctx = canvas.getContext("2d");
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const setDim = document.getElementById("setDim");
    const clearButton = document.getElementById("clearButton");
    const generar = document.getElementById("generar");
    const dimInput = document.getElementById("dimInput");
    const agregar = document.getElementById("agregar");
    const res = document.getElementById("res");
    let drawing = false;
    let datos = [];

    function esPotenciaDe2(n) {
        return n > 0 && (n & (n - 1)) === 0;
    }
    function ajusta(img){
        // la imagen stiene que ser mayor o igual a 32x32 y cuadrada pero tambien una potencia de 2
        let [x,y]= dimension(img);
        if(x>32 && y==x && esPotenciaDe2(x) ){
            const factor= dimension(img)[0]/32;
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
    clearButton.addEventListener("click", () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      res.innerHTML = "?";
    });
    setDim.addEventListener("click", () => {
      const dimensions = parseInt(dimInput.value, 10);
      const MAX_DIM = 512;
      if (!isNaN(dimensions) && dimensions > 0 && dimensions <= MAX_DIM && esPotenciaDe2(dimensions) && dimensions >=32) {
        canvas.width = dimensions;
        canvas.height = dimensions;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      } else {
        alert(`Por favor ingresa un número válido entre 32-512 y que sea potencia de 2.`);
      }
    });
    generar.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(datos)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "canvas_matrix.json";
    link.click();
    URL.revokeObjectURL(url); // 🔥 libera memoria
    console.log("se ha descargado un dataset de", datos.length, "imagenes");
    });
    agregar.addEventListener("click", () => {
    datos.push({ inp: canvasToMatrix() });
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    console.log("agregado", datos.length);
    });
    predecir.addEventListener("click", () => {
    let img = canvasToMatrix();
    img = ajusta(img);
    img = normalizeSample(img);
    res.innerHTML = evalBatchPreds(img, model);
    });
    
});