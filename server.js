// server.js
const express = require('express');
const { spawn } = require('child_process');
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
    res.sendFile(__dirname + '/index.html');
});

// API to run the Python model and send predictions
app.get('/predict', (req, res) => {
    const python = spawn('python', ['sign_language_recognition.py']);
    
    python.stdout.on('data', (data) => {
        res.send(data.toString());
    });

    python.stderr.on('data', (data) => {
        console.error(`Error: ${data}`);
    });
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});