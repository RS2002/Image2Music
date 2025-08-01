from music21 import converter, stream
import muspy
import copy

def generation_parse(s): # parse the generated result of VLM
    start_index = s.rfind('X:')
    end_index = s.rfind('|')
    if start_index == -1 or end_index == -1:
        print("Text Parse Fail")
        return None, None
    abc_music = s[start_index:end_index + 1]
    description = s[end_index + 1:]
    return abc_music, description

def parse_abc(abc_str):
    track_index = [i for i, c in enumerate(abc_str) if (c == 'v' or c == "V")]
    if len(track_index) <= 1:  # single track
        return converter.parse(abc_str, format='abc')
    else:
        head = abc_str[:track_index[0]]
        score = stream.Score()

        # 预先解析头部获取全局元数据
        global_metadata = converter.parse(head, format='abc')

        for i in range(len(track_index)):
            start_pos = track_index[i]
            if i == len(track_index) - 1:
                end_pos = -1
            else:
                end_pos = track_index[i + 1]
            part_abc = head + abc_str[start_pos:end_pos]
            part_stream = converter.parse(part_abc, format='abc')

            # 创建独立的声部并复制元素
            new_part = stream.Part()

            # 复制全局元数据（深拷贝）
            for elem in global_metadata.recurse().getElementsByClass(['TimeSignature', 'KeySignature']):
                new_part.insert(0, copy.deepcopy(elem))

            # 复制音符元素
            for elem in part_stream.recurse().notesAndRests:
                new_part.append(copy.deepcopy(elem))

            score.append(new_part)

        return score

def evaluate(music):
    pitch_range = muspy.pitch_range(music)
    n_pitches_used = muspy.n_pitches_used(music)
    n_pitch_classes_used = muspy.n_pitch_classes_used(music)
    polyphony = float(muspy.polyphony(music))
    scale_consistency = muspy.scale_consistency(music)
    pitch_entropy = float(muspy.pitch_entropy(music))
    pitch_class_entropy = float(muspy.pitch_class_entropy(music))
    empty_beat_rate = muspy.empty_beat_rate(music)
    return [pitch_range, n_pitches_used, n_pitch_classes_used, polyphony, scale_consistency, pitch_entropy, pitch_class_entropy, empty_beat_rate]