import subprocess
import psutil
import time
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

data = {
    "time": [],
    "cpu": [],
    "ram": [],
    "temp": []
}

start_time = time.time()

# Get CPU temperature using vcgencmd
def get_cpu_temp():
    try:
        result = subprocess.run(
            ['vcgencmd', 'measure_temp'],
            capture_output=True,
            text=True,
            timeout=1
        )

        temp_str = result.stdout.strip()
        temp = float(temp_str.split("=")[1].replace("'C",""))
        return temp
    except:
        return 0


@app.route("/stats")
def stats():

    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    temp = get_cpu_temp()

    current_time = round(time.time() - start_time, 1)

    data["time"].append(current_time)
    data["cpu"].append(cpu)
    data["ram"].append(ram)
    data["temp"].append(temp)

    if len(data["time"]) > 200:
        for k in data:
            data[k].pop(0)

    return jsonify(data)


@app.route("/")
def index():
    return render_template_string("""
<html>
<head>
<title>Raspberry Pi Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>

<body>
<h2>Raspberry Pi System Monitor</h2>
<canvas id="chart" width="900" height="400"></canvas>

<script>

const ctx = document.getElementById('chart').getContext('2d');

const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {label: "CPU %", data: [], borderColor: "red"},
            {label: "RAM %", data: [], borderColor: "blue"},
            {label: "Temp °C", data: [], borderColor: "green"}
        ]
    },
    options: {
        animation:false,
        scales: {
            y: {beginAtZero: true}
        }
    }
});

async function update(){

    const res = await fetch("/stats");
    const data = await res.json();

    chart.data.labels = data.time;
    chart.data.datasets[0].data = data.cpu;
    chart.data.datasets[1].data = data.ram;
    chart.data.datasets[2].data = data.temp;

    chart.update();
}

setInterval(update, 1000);

</script>
</body>
</html>
""")

if __name__ == "__main__":
    print("Starting Raspberry Pi Monitor")
    print("Open browser at: http://<pi-ip>:5000")
    app.run(host="0.0.0.0", port=5000)