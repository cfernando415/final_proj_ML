document.addEventListener("DOMContentLoaded", () => {
    fetch("http://localhost:3000/api/v1/diseases")
        .then(res => res.json())
        .then(data => {
            let x_axis = data.map(el => { return el.year });
            let y_axis = data.map(el => { return el.deaths });
            dataHandler(x_axis, y_axis, data);
        })
})

function dataHandler(x_vals, y_vals, data_arr) {

    const learningRate = 0.0042;
    const optimizer = tf.train.sgd(learningRate);
    const n = data_arr.length;
    const sumOfX = x_vals.reduce((a, b) => a + b, 0);
    const sumOfY = y_vals.reduce((a, b) => a + b, 0);
    const sumOfX2 = x_vals.map(el => el**2).reduce((a, b) => a + b, 0);
    const sumOfXY = data_arr.map(el => el.year * el.deaths).reduce((a, b) => a + b, 0);

    const slope = findSlope(n, sumOfX, sumOfY, sumOfX2, sumOfXY);
    
    const yIntercept = findYIntercept(n, sumOfX, sumOfY, slope);
    
    let m = tf.variable(tf.scalar(slope));
    
    let b = tf.variable(tf.scalar(yIntercept));

    tf.tidy(() => {
        if(x_vals.length > 0) {
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(x_vals, m, b), ys));
        }
    });
    

    const trace1 = {
        x: x_vals,
        y: y_vals,
        mode: 'markers+Text',
        type: 'scatter',
        name: 'Male',
        marker: {
            color: 'rgb(219, 64, 82)'
        }
    };

    const lineX = [Math.min.apply(null, x_vals), Math.max.apply(null, x_vals)];

    const ys = tf.tidy(() => predict(lineX, m, b));
    lineY = ys.dataSync();

    ys.dispose();
   
    const trace2 = {
        x: lineX,
        y: lineY,
        name: 'trendline',
        mode: 'lines',
        type: 'scatter'
    };

    let layout = {
        autosize: true,
        width: 500,
        height: 500,
        margin: {
          l: 50,
          r: 50,
          b: 100,
          t: 100,
          pad: 4
        },
      };
    
    let data = [trace1, trace2];

    Plotly.newPlot('tester', data, layout);

    
}

function predict(x_args, m, b) {

    const xs = tf.tensor1d(x_args);
    
    const ys = xs.mul(m).add(b); // y = mx + b

    return ys;
}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}

function findSlope(n, sumOfX, sumOfY, sumOfX2, sumOfXY) {
    return (n*sumOfXY - sumOfX*sumOfY)/(n*sumOfX2 - sumOfX**2);
}

function findYIntercept(n, sumOfX, sumOfY, slope) {
    return (sumOfY - (slope * sumOfX))/n;
}