from gtts import gTTS
import os
text = "Chú ý. Bạn có vẻ đã mệt mỏi. Hãy nghỉ ngơi"
tts = gTTS(text=text, lang='vi')
output_path = "assets/yawning.mp3"
# Lưu file (ghi đè file cũ)
tts.save(output_path)