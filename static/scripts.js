// static/scripts.js

document.addEventListener('DOMContentLoaded', function() {
    const processBtn = document.getElementById('process-btn');
    const saveBtn = document.getElementById('save-btn');
    const resultImage = document.getElementById('result-image');
    const parametersForm = document.getElementById('parameters-form');

    processBtn.addEventListener('click', function() {
        // สร้าง FormData จากแบบฟอร์มพารามิเตอร์
        const formData = new FormData(parametersForm);
        const params = {};

        formData.forEach((value, key) => {
            // แปลงค่าต่าง ๆ ให้เป็นชนิดข้อมูลที่ถูกต้อง
            if (key === 'invert' || key === 'sharpen' || key === 'edge_enhance' || key === 'hist_eq' || key === 'clahe' || key === 'local_contrast' || key === 'screen_tone_1' || key === 'screen_tone_2') {
                params[key] = value === 'on' || value === 'true' ? true : false;
            } else if (key === 'screen_tone_color_1' || key === 'screen_tone_color_2') {
                params[key] = value; // ค่าสีเป็นสตริง เช่น "#323232"
            } else {
                // แปลงค่าตัวเลข
                const numValue = Number(value);
                params[key] = isNaN(numValue) ? value : numValue;
            }
        });

        // ส่งคำขอ AJAX ไปยัง '/process'
        fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'done') {
                resultImage.src = 'data:image/png;base64,' + data.image;
            } else {
                alert('เกิดข้อผิดพลาดในการประมวลผลภาพ');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('เกิดข้อผิดพลาดในการประมวลผลภาพ');
        });
    });

    saveBtn.addEventListener('click', function() {
        // ตรวจสอบว่ามีภาพที่ประมวลผลอยู่หรือไม่
        if (resultImage.src) {
            window.location.href = '/save_image';
        } else {
            alert('ยังไม่มีภาพที่ประมวลผลให้บันทึก');
        }
    });
});
