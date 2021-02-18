"""
    Visual-Template-free-Form-Parsting is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Visual-Template-free-Form-Parsting is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Visual-Template-free-Form-Parsting.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import sys
def str2label_single(value, characterToIndex={}, unknown_index=None):
    if unknown_index is None:
        unknown_index = len(characterToIndex)

    label = []
    for v in value:
        if v not in characterToIndex:
            continue
            # raise "Unknown Charactor to Label conversion"
        label.append(characterToIndex[v])
    return np.array(label, np.uint32)

def label2input_single(value, num_of_inputs, char_break_interval):
    idx1 = len(value) * (char_break_interval + 1) + char_break_interval
    idx2 = num_of_inputs + 1
    input_data = [[0 for i in range(idx2)] for j in range(idx1)]

    cnt = 0
    for i in range(char_break_interval):
        input_data[cnt][idx2-1] = 1
        cnt += 1

    for i in range(len(value)):
        if value[i] == 0:
            input_data[cnt][idx2-1] = 1
        else:
            input_data[cnt][value[i]-1] = 1
        cnt += 1

        for i in range(char_break_interval):
            input_data[cnt][idx2-1] = 1
            cnt += 1

    return np.array(input_data)

def label2str_single(label, indexToCharacter, asRaw, spaceChar = "~"):
    string = u""
    for i in range(len(label)):
        if label[i] == 0:
            if asRaw:
                string += spaceChar
            else:
                break
        else:
            val = label[i]
            string += indexToCharacter[val]
    return string

def naive_decode(output):
    rawPredData = np.argmax(output, axis=1)
    predData = []
    for i in range(len(output)):
        if rawPredData[i] != 0 and not ( i > 0 and rawPredData[i] == rawPredData[i-1] ):
            predData.append(rawPredData[i])
    return predData, list(rawPredData)
