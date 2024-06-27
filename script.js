document.getElementById('sendButton').addEventListener('click', function() {
    const url = document.getElementById('link').value;
    console.log(url)
    fetch('http://127.0.0.1:8000/api/run-code/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: url })
    })
    .then(response => {
        // Check if the response has the correct content type
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('Unexpected response content type');
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('result').innerText = `${data.data}`;
    })
    .catch((error) => {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'An error occurred.';
    });
});
