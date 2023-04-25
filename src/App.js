import './App.css';
import React, { useRef } from "react";
import CanvasDraw from "react-canvas-draw";
import * as tf from '@tensorflow/tfjs';

const App = () => {

  const canvasRef = useRef();
  const [modelo_convu, setModelo] = React.useState([]);
  const [modelo_denso, setModeloDenso] = React.useState([]);
  React.useEffect(() => {
    async function loadModel() {
      const model_convu = await tf.loadLayersModel('./model.json');
      const model_dense = await tf.loadLayersModel('./model_dense.json');

      setModelo(model_convu);
      setModeloDenso(model_dense);
    }   
    loadModel();
  }, []);

  function predecir() {

    //Pasar canvas a version 28x28
    
    // create  small temporal canvas 

    var smallcanvas = document.createElement("canvas");
    smallcanvas = smallcanvas.getContext("2d"); // Get the 2D context

    smallcanvas.width = 28;
    smallcanvas.height = 28;


    resample_single(canvasRef, 28, 28, smallcanvas);
    
    var imgData = smallcanvas.getImageData(0,0,28,28);
    
    var arr = []; //El arreglo completo
    
    var arr28 = []; //Al llegar a 28 posiciones se pone en 'arr' como un nuevo indice
    
    for (var p=0, i=0; p < imgData.data.length; p+=4) {
    
    var valor = imgData.data[p+3]/255;
    
    arr28.push([valor]); //Agregar al arr28 y normalizar a 0-1. Aparte queda dentro de un arreglo en el indice 0... again
    
    if (arr28.length == 28) {
    
    arr.push(arr28);
    
    arr28 = [];
    
    }
    
    }
    
    arr = [arr]; //Meter el arreglo en otro arreglo por que si no tio tensorflow se enoja >:(
    
    var tensor4 = tf.tensor4d(arr);
    
    var resultado_convo = modelo_convu.predict(tensor4).dataSync();
    var resultado_denso = modelo_denso.predict(tensor4).dataSync(); 

    var mayorIndiceConvo = resultado_convo.indexOf(Math.max.apply(null, resultado_convo));
    var mayorIndiceDenso = resultado_denso.indexOf(Math.max.apply(null, resultado_denso));
    
    document.getElementById("resultado").innerHTML = 
    "Preddicion denso: " + mayorIndiceDenso +
    " Preddicion convolucional: " + mayorIndiceConvo;

    }
  function resample_single(canvas, width, height, resize_canvas) {

    canvas = canvas.current.canvas.drawing

    var width_source = canvas.width;
    
    var height_source = canvas.height;
    
    width = Math.round(width);
    
    height = Math.round(height);
    
    var ratio_w = width_source / width;
    
    var ratio_h = height_source / height;
    
    var ratio_w_half = Math.ceil(ratio_w / 2);
    
    var ratio_h_half = Math.ceil(ratio_h / 2);
    
    var ctx = canvas.getContext("2d");
    
    var ctx2 = resize_canvas;
    
    var img = ctx.getImageData(0, 0, width_source, height_source);
    
    var img2 = ctx2.createImageData(width, height);
    
    var data = img.data;
    
    var data2 = img2.data;
    
    for (var j = 0; j < height; j++) {
    
    for (var i = 0; i < width; i++) {
    
    var x2 = (i + j * width) * 4;
    
    var weight = 0;
    
    var weights = 0;
    
    var weights_alpha = 0;
    
    var gx_r = 0;
    
    var gx_g = 0;
    
    var gx_b = 0;
    
    var gx_a = 0;
    
    var center_y = (j + 0.5) * ratio_h;
    
    var yy_start = Math.floor(j * ratio_h);
    
    var yy_stop = Math.ceil((j + 1) * ratio_h);
    
    for (var yy = yy_start; yy < yy_stop; yy++) {
    
    var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
    
    var center_x = (i + 0.5) * ratio_w;
    
    var w0 = dy * dy; //pre-calc part of w
    
    var xx_start = Math.floor(i * ratio_w);
    
    var xx_stop = Math.ceil((i + 1) * ratio_w);
    
    for (var xx = xx_start; xx < xx_stop; xx++) {
    
    var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
    
    var w = Math.sqrt(w0 + dx * dx);
    
    if (w >= 1) {
    
    continue;
    
    }
    
    weight = 2 * w * w * w - 3 * w * w + 1;
    
    var pos_x = 4 * (xx + yy * width_source);
    
    gx_a += weight * data[pos_x + 3];
    
    weights_alpha += weight;
    
    //colors
    
    if (data[pos_x + 3] < 255)
    
    weight = weight * data[pos_x + 3] / 250;
    
    gx_r += weight * data[pos_x];
    
    gx_g += weight * data[pos_x + 1];
    
    gx_b += weight * data[pos_x + 2];
    
    weights += weight;
    
    }
    
    }
    
    data2[x2] = gx_r / weights;
    
    data2[x2 + 1] = gx_g / weights;
    
    data2[x2 + 2] = gx_b / weights;
    
    data2[x2 + 3] = gx_a / weights_alpha;
    
    }
    
    }
    
    for (var p=0; p < data2.length; p += 4) {
    
    var gris = data2[p]; //Esta en blanco y negro
    
    if (gris < 100) {
    
    gris = 0; //exagerarlo
    
    } else {
    
    gris = 255; //al infinito
    
    }
    
    data2[p] = gris;
    
    data2[p+1] = gris;
    
    data2[p+2] = gris;
    
    }
    ctx2.putImageData(img2, 0, 0);
    
    }
  const handleEraseButtonClick = () => {
    if (canvasRef.current) {
      canvasRef.current.eraseAll();
    }
  };
  const handleSaveClick = () => {
    if (canvasRef.current) {
      let image = canvasRef.current.getDataURL();
      window.open(image);
      return canvasRef.current.getDataURL();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <CanvasDraw 
        ref={canvasRef}
        id = "canvasDraw"
       hideInterface
       hideGrid 
       brushRadius={4}
        />
        <div class="buttons-container">
      <button
      onClick={handleEraseButtonClick}
          >
            Borrar
          </button>
          <button
            onClick={() => {
              predecir()
            }}
          >
            Predecir
          </button></div>
    
        <div id="resultado"></div>
      </header>
    </div>
  );
}

export default App;

