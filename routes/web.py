from flask import Blueprint, render_template
import config

web_bp = Blueprint('web', __name__)

@web_bp.route('/', methods=['GET'])
def index():
    return render_template('index.html',
                           threshold_methods=config.THRESHOLD_METHODS,
                           noise_reduction_methods=config.NOISE_REDUCTION_METHODS,
                           screen_tone_patterns=config.SCREEN_TONE_PATTERNS,
                           screen_tone_area_patterns=config.SCREEN_TONE_AREA_PATTERNS,
                           pencil_shading_styles=config.PENCIL_SHADING_STYLES)
