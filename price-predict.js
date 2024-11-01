const { spawn } = require('child_process');

//const childPython = spawn('python', ['--version']);
// const childPython2 = spawn('python', ['backend.py', 'Area', 'No. of Bedrooms', 'City', 'Location', 'Resale']);
const childPython = spawn('python', ['backend.py', '1675', '3', 'Banglore', 'Doddanekundi', '0', 'Random Forest']);

childPython.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
});

childPython.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
});

childPython.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
}); 