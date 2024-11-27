document.getElementById('process-btn').addEventListener('click', function() {
    // เก็บค่าพารามิเตอร์จากฟอร์ม
    const formData = new FormData(document.getElementById('parameters-form'));
    const params = {};
    for (let [key, value] of formData.entries()) {
        const element = document.getElementById(key);
        if (element.type === 'checkbox') {
            params[key] = element.checked.toString();
        } else {
            params[key] = value;
        }
    }

    // ส่งพารามิเตอร์การประมวลผลไปยังเซิร์ฟเวอร์
    fetch('/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    }).then(response => response.json()).then(data => {
        if (data.status === 'done') {
            document.getElementById('result-image').src = 'data:image/png;base64,' + data.image;
        } else {
            alert('เกิดข้อผิดพลาดในการประมวลผลภาพ');
        }
    });
});

document.getElementById('save-btn').addEventListener('click', function() {
    window.location.href = '/save_image';
});
