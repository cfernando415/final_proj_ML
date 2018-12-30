document.addEventListener("DOMContentLoaded", () => {
    fetch("http://localhost:3000/api/v1/diseases")
        .then(res => res.json())
        .then(data => {
            const x_axis = data.map(el => { return el.year });
            const y_axis = data.map(el => { return el.deaths });
            dataHandler(x_axis, y_axis, data);
            loadNavigation(data);
        })
    document.getElementById("nav").addEventListener("click", navHandler);
})

// let a, b, c, d;
// let data_x, data_y;

// var cubic = function(params,x) {
//     return params[0] * x*x*x + params[1] * x*x + params[2] * x + params[3];
// };

// var objective = function(params) {
//     var total = 0.0;
//     for(var i=0; i < data_x.length; ++i) {
//       var resultThisDatum = cubic(params, data_x[i]);
//       var delta = resultThisDatum - data_y[i];
//       total += (delta*delta);
//     }
//     return total;
// };

function dataHandler(x_vals, y_vals, data_arr) {
    
    const tmp_arr = data_arr.map(el => [el.year, el.deaths])
    // const tmp = data_arr.filter(el => el.leading_cause === "Diseases of Heart (I00-I09, I11, I13, I20-I51)" && (el.sex ==='M' || el.sex === "Male"));

    // var initial = [1,1,1,1];
    // data_x = x_vals;
    // data_y = y_vals;

    // var minimiser = numeric.uncmin(objective,initial);

    // const learningRate = 0.0042;
    const learningRate = 0.00000001;
    const optimizer = tf.train.adam(learningRate);

    const n = data_arr.length;
    const sumOfX = x_vals.reduce((x, j) => x + j, 0);
    const sumOfY = y_vals.reduce((x, j) => x + j, 0);
    const sumOfX2 = x_vals.map(el => el**2).reduce((x, j) => x + j, 0);
    const sumOfXY = data_arr.map(el => el.year * el.deaths).reduce((x, j) => x + j, 0);

    const slope = findSlope(n, sumOfX, sumOfY, sumOfX2, sumOfXY);
    
    const yIntercept = findYIntercept(n, sumOfX, sumOfY, slope);
    
    let m = tf.variable(tf.scalar(slope));
    
    let b = tf.variable(tf.scalar(yIntercept));

    // a = tf.variable(tf.scalar(minimiser.solution[0]));
    // b = tf.variable(tf.scalar(minimiser.solution[1]));
    // c = tf.variable(tf.scalar(minimiser.solution[2]));
    // d = tf.variable(tf.scalar(minimiser.solution[3]));

    // a = tf.variable(tf.scalar(1));
    // b = tf.variable(tf.scalar(2));
    // c = tf.variable(tf.scalar(1));
    // d = tf.variable(tf.scalar(0));

    tf.tidy(() => {
        if(x_vals.length > 0) {
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(x_vals, m, b), ys));
            // optimizer.minimize(() => loss(predict(x_vals), ys));
        }
    });

    const trace1 = {
        x: data_arr.filter(el => el.sex === 'M' || el.sex === 'Male').map(el => el.year),
        y: data_arr.filter(el => el.sex === 'M' || el.sex === 'Male').map(el => el.deaths),
        mode: 'markers+Text',
        type: 'scatter',
        name: 'Male',
        marker: {
            color: 'rgb(36,	98,	175)'
        }
    };

    // const lineX = [Math.min.apply(null, x_vals), (Math.max.apply(null, x_vals))];
    const lineX = x_vals.filter((v, i, a) => a.indexOf(v) === i)
    const ys = tf.tidy(() => predict(lineX, m, b));
    // const ys = tf.tidy(() => predict(lineX));
    lineY = ys.dataSync();
    ys.dispose();

    const trace2 = {
        x: lineX,
        y: lineY,
        name: 'trendline',
        mode: 'lines',
        type: 'scatter',
        // connectgaps: true
    };

    const trace3 = {
        x: data_arr.filter(el => el.sex === 'F' || el.sex === 'Female').map(el => el.year),
        y: data_arr.filter(el => el.sex === 'F' || el.sex === 'Female').map(el => el.deaths),
        mode: 'markers+Text',
        type: 'scatter',
        name: 'Female',
        marker: {
            color: 'rgb(233, 121, 127)'
        }
    };
    
    const chartTitle = (data_arr.map(e => e.leading_cause).filter((v, i, a) => a.indexOf(v) === i).length > 1) ? "All Diseases" : data_arr.map(e => e.leading_cause).filter((v, i, a) => a.indexOf(v) === i)[0]; 

    let layout = {
        title: chartTitle,
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
    
    let data = [trace1, trace2, trace3];

    Plotly.newPlot('tester', data, layout, {responsive: true});


}

function predict(x_args, m, b) {

    const xs = tf.tensor1d(x_args);
    
    const ys = xs.mul(m).add(b); // y = mx + b
    // const ys = xs.pow(tf.scalar(3)).mul(a)
    //     .add(xs.square().mul(b))
    //     .add(xs.mul(c))
    //     .add(d);

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

function loadNavigation(arg) {
    const divRoot = document.getElementById("nav");
    const categories = arg.map(el => el.leading_cause).filter((x, i, a) => a.indexOf(x) == i);

    for(el of categories) {
        const divNode = document.createElement("div");
        divNode.className = "category div_hover";
        divNode.innerText = el;
        divRoot.appendChild(divNode);
    }
    
}

function navHandler(e) {
    const categoryName = e.target.innerText;

    if(e.target.classList.contains("category")){
        fetch("http://localhost:3000/api/v1/diseases")
            .then(res => res.json())
            .then((json) => {              
                const filteredJSON = json.filter(el => el.leading_cause === categoryName);
                const x_axis = filteredJSON.map(el => { return el.year });
                const y_axis = filteredJSON.map(el => { return el.deaths });
                dataHandler(x_axis, y_axis, filteredJSON);
            })
    }
}