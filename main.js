document.addEventListener('DOMContentLoaded', () => {

    setupUI();

    const commonPreviewArea = document.getElementById('common-card-preview');
    
    if (commonPreviewArea) {
        commonPreviewArea.addEventListener('click', () => {
            const previewImg = commonPreviewArea.querySelector('img');
            const mainImgSrc = (previewImg && previewImg.src) ? previewImg.src : null;
            const flavor1Src = commonPreviewArea.dataset.flavor1;
            const flavor2Src = commonPreviewArea.dataset.flavor2;

            if (typeof lightboxContent !== 'undefined' && typeof lightboxOverlay !== 'undefined') {
                lightboxContent.innerHTML = '';
                let imagesAdded = 0;
                [mainImgSrc, flavor1Src, flavor2Src].forEach(src => {
                    if (src) {
                        const img = document.createElement('img');
                        img.src = src;
                        lightboxContent.appendChild(img);
                        imagesAdded++;
                    }
                });
                if (imagesAdded > 0) {
                    lightboxOverlay.classList.add('show');
                }
            }
        });
    }

    initializeBoard('.player-wrapper', '');
    initializeBoard('.opponent-wrapper', 'opponent-');

});
