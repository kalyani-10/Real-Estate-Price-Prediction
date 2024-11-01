const express = require('express');
const bodyParser = require('body-parser');

const app = express();
const port = 3000;

app.use(bodyParser.urlencoded({ extended: false }));
app.use(express.static('public'));

app.post('/', (req, res) => {
    const citySelection = req.body.citySelection;
    const localitySelection = req.body.localitySelection;
    const BHK = req.body.BHK;
    const model = req.body.model;
    const Area = req.body.Area;
    const Resell = req.body.Resell;
    const output = `You entered: ${citySelection}, ${localitySelection}, ${BHK}, ${model}, ${Area}, ${Resell}`;
    res.send(output);
});

app.listen(port, () => {
    console.log(`Server started on port ${port}`);
});

//////////////////

const price = document.getElementById('price');
const output = document.getElementById('output');

form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const citySelection = form.elements.citySelection.value;
    const localitySelection = form.elements.localitySelection.value;
    const BHK = form.elements.BHK.value;
    const model = form.elements.model.value;
    const Area = form.elements.Area.value;
    const Resell = form.elements.Resell.value;

    const response = await fetch('/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `citySelection=${encodeURIComponent(citySelection)}, localitySelection=${encodeURIComponent(localitySelection)}, 
        BHK=${encodeURIComponent(BHK)}, model=${encodeURIComponent(model)}, Area=${encodeURIComponent(Area)}, Resell=${encodeURIComponent(Resell)}`
    });

    const outputText = await response.text();

    output.textContent = outputText;
});

