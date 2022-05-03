window.addEventListener("load", ()=> {
  const canvas = document.querySelector("#canvas");
  const ctx = canvas.getContext("2d");

  // canvas.height = window.innerHeight * 0.5;
  // canvas.width = window.innerWidth * 0.5;
  canvas.height = 280;
  canvas.width = 280;
  
  let drawing = false;

  function start({clientX: x, clientY: y}) {
    drawing = true;
    ctx.beginPath();
    
    draw({clientX: x, clientY: y});
  }
  function stop() {
    drawing = false;
    
  }
  function draw({clientX: x, clientY: y}) {
    if (!drawing) {
      return;
    } 
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.lineTo(x-5, y-5);
    ctx.stroke();
  }


  window.addEventListener("mousedown", start);
  window.addEventListener("mouseup", stop);
  window.addEventListener("mousemove", draw)

  
});

// window.addEventListener("resize", ()=> {
//   canvas.height = window.innerHeight * 0.5;
//   canvas.width = window.innerWidth * 0.5;
// });

const resetCanvas = function() {
  const canvas = document.querySelector("#canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

document.getElementById("reset").addEventListener("click",resetCanvas);

// const updateData = function() {
//   const canvas = document.querySelector("#canvas");
//   document.getElementById("data-paint").value = canvas.toDataURL("image/png")
  
// }

const updateData = function() {
  const canvas = document.querySelector("#canvas");
  const digit = canvas.toDataURL("image/png");
  // alert(digit);
  // alert(typeof digit);
  // $("data-paint").val((digit));
  // alert(JSON.stringify([
  //   {digit}
  // ]));


  $.ajax({
    type: "POST",
    url: "/digit",
    // data: digit,
    data: JSON.stringify(
        {digit}
      ),
    contentType: "application/json",
    dataType: "json",
    // success: function(comeback) {
    //   alert(comeback);
    // }
  })
  .then((prediction) => {
    // alert("success");
    $(`<h1>${prediction}</h1>`).appendTo("body");
  })
  .catch(() => {
    alert("failed");
  })
  
}

// const submitData = function() {
//   const canvas = document.querySelector("#canvas");
//   $.ajax({
//     type: "POST",
//     url: "http://localhost:5000/data",
//     // data: {digit: $("#canvas").toDataURL()},
//     data: {
//       imageBase64: canvas.toDataURL()
//     }
    
//   }).done(function() {
//     console.log('sent');
//   })
  
  
// }

document.getElementById("submit").addEventListener("click",updateData)

//document.getElementById("submit").addEventListener("click",submitData)
