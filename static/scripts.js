// static/scripts.js

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const processBtn = document.getElementById('process-btn');
    const saveBtn = document.getElementById('save-btn');
    const resultImage = document.getElementById('result-image');
    const parametersForm = document.getElementById('parameters-form');

    // ฟังก์ชันสำหรับส่งพารามิเตอร์และประมวลผลภาพ
    function processImage(params) {
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
    }

    // การจัดการการอัปโหลดภาพ
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault(); // ป้องกันการรีเฟรชหน้า

        const formData = new FormData(uploadForm);

        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(data => {
            // อัปเดตหน้าเว็บด้วยข้อมูลที่ได้รับจากเซิร์ฟเวอร์
            document.body.innerHTML = data;

            // หลังจากอัปโหลดภาพแล้ว ให้ทำการประมวลผลภาพโดยอัตโนมัติด้วยค่าพารามิเตอร์เริ่มต้น
            const defaultParams = {
                threshold: 128,
                contrast: 1.0,
                brightness: 1.0,
                gamma: 1.0,
                exposure: 1.0,
                method: 'Special Adaptive Thresholding',
                block_size: 11,
                c_value: 6,
                invert: false,
                noise_reduction: 'None',
                sharpen: false,
                edge_enhance: false,
                hist_eq: false,
                clahe: false,
                clip_limit: 2.0,
                tile_grid_size: 8,
                local_contrast: false,
                kernel_size: 9,
                screen_tone_1: false,
                screen_tone_pattern_1: 'None',
                screen_tone_size_1: 50,
                screen_tone_density_1: 50,
                screen_tone_area_pattern_1: 'Global Darkness',
                pencil_shading_style_1: 'Light',
                screen_tone_color_1: '#323232',
                screen_tone_2: false,
                screen_tone_pattern_2: 'None',
                screen_tone_size_2: 50,
                screen_tone_density_2: 70,
                screen_tone_area_pattern_2: 'Shadow Regions',
                pencil_shading_style_2: 'Medium',
                screen_tone_color_2: '#505050'
            };

            processImage(defaultParams);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('เกิดข้อผิดพลาดในการอัปโหลดภาพ');
        });
    });

    // การจัดการปุ่ม "ประมวลผลภาพ"
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

        // ส่งคำขอประมวลผลภาพ
        processImage(params);
    });

    // การจัดการปุ่ม "บันทึกภาพ"
    saveBtn.addEventListener('click', function() {
        if (resultImage.src) {
            window.location.href = '/save_image';
        } else {
            alert('ยังไม่มีภาพที่ประมวลผลให้บันทึก');
        }
    });
});
