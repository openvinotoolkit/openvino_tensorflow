const combination = new Map([
    ["000000", "# For Python3.7 and Python3.9 change the versions in the links appropriately , pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/tensorflow_abi1-2.8.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl , pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/openvino_tensorflow_abi1-2.0.0-cp38-cp38-manylinux_2_27_x86_64.whl , Download Intel® Distribution of OpenVINO™ Toolkit Link-https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?operatingsystem=linux&distributions=webdownload&version=2021%203%20(latest)&options=offline , source $INTEL_OPENVINO_DIR/setupvars.sh"],
    ["000100", "# For Python3.7 and Python3.9 change the versions in the links appropriately , pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/tensorflow_abi1-2.8.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl , pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/openvino_tensorflow_abi1-2.0.0-cp38-cp38-manylinux_2_27_x86_64.whl , Download Intel® Distribution of OpenVINO™ Toolkit Link-https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?operatingsystem=linux&distributions=webdownload&version=2021%203%20(latest)&options=offline , source $INTEL_OPENVINO_DIR/setupvars.sh"],
    ["001000", " Windows not yet supported for VAD-M"],
    ["001100", " Mac OS not yet supported for VAD-M"],
    ["000001", " pip3 install -U pip,pip3 install tensorflow==2.8.0, pip3 install openvino-tensorflow==2.0.0"],
    ["000101", " pip3 install -U pip,pip3 install tensorflow==2.8.0,pip3 install openvino-tensorflow==2.0.0"],
    ["001001", " pip3 install -U pip,pip3 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/tensorflow-2.8.0-cp39-cp39-win_amd64.whl, pip3 install openvino-tensorflow==2.0.0"],
    ["001101", " pip3 install -U pip,pip3 install tensorflow==2.8.0,pip3 install openvino-tensorflow==2.0.0"],
    ["010001", " pip3 install openvino-tensorflow==2.0.0"],
    ["010101", " pip3 install openvino-tensorflow==2.0.0"],
    ["011001", " OpenVINO™ integration with TensorFlow doesn't supports PyPi TensorFlow"],
    ["011101", " pip3 install openvino-tensorflow==2.0.0"],
    ["100000", "# For Python3.7 and Python3.9 change the versions in the links appropriately , pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/tensorflow_abi1-2.8.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl, pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/openvino_tensorflow_abi1-2.0.0-cp38-cp38-manylinux_2_27_x86_64.whl, source $INTEL_OPENVINO_DIR/setupvars.sh"],
    ["100100", "# For Python3.7 and Python3.9 change the versions in the links appropriately , pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/tensorflow_abi1-2.8.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl , pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/openvino_tensorflow_abi1-2.0.0-cp38-cp38-manylinux_2_27_x86_64.whl , source $INTEL_OPENVINO_DIR/setupvars.sh"],
    ["101000", " Windows not yet supported for VAD-M"],
    ["101100", " Mac OS not yet supported for VAD-M"],
    ["000010", " Download Intel® Distribution of OpenVINO™ Toolkit Link-https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?operatingsystem=linux&distributions=webdownload&version=2021%203%20(latest)&options=offline ,python3 build_ovtf.py --use_openvino_from_location=$INTEL_OPENVINO_DIR --cxx11_abi_version=1"],
    ["000110", " Download Intel® Distribution of OpenVINO™ Toolkit Link-https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?operatingsystem=linux&distributions=webdownload&version=2021%203%20(latest)&options=offline ,python3 build_ovtf.py --use_openvino_from_location=$INTEL_OPENVINO_DIR --cxx11_abi_version=1"],
    ["001010", " Windows not yet supported for VAD-M"],
    ["001110", " Mac OS not yet supported for VAD-M"],
    ["000011", " python3 build_ovtf.py "],
    ["000111", " python3 build_ovtf.py "],
    ["001011", " Build from source on Windows needs Intel® Distribution of OpenVINO™ Toolkit , python build_ovtf.py --tf_version=v2.8.0 --use_openvino_from_location=$INTEL_OPENVINO_DIR "],
    ["001111", " python3 build_ovtf.py"],
    ["100010", " #Use OpenVINO 2022.1.0,python3 build_ovtf.py --use_openvino_from_location=$INTEL_OPENVINO_DIR --cxx11_abi_version=1"],
    ["100110", " #Use OpenVINO 2022.1.0,python3 build_ovtf.py --use_openvino_from_location=$INTEL_OPENVINO_DIR --cxx11_abi_version=1"],
    ["101010", " Windows not yet supported for VAD-M"],
    ["101110", " Mac OS not yet supported for VAD-M"],
    ["100011", " python3 build_ovtf.py"],
    ["100111", " python3 build_ovtf.py"],
    ["101011", " python3 build_ovtf.py --use_openvino_from_location=$INTEL_OPENVINO_DIR"],
    ["101111", " python3 build_ovtf.py"]
]);

window.onload = new function() {
    assignfunction();
    commandArea();

};

function assignfunction() {
    const divs = document.querySelectorAll('div');

    for (let div of divs) {
        if (div.id.startsWith('option')) {
            div.onclick = () => {
                if (div.classList.contains('selected')) {
                    div.classList.remove('selected');
                } else {
                    div.classList.add('selected');
                }
                removeSameRow(div);
                removeFromOtherRow(div);
                commandArea();
            }
        }
    }
}

function removeFromOtherRow(div) {
    if (div.id == 'option-byo-tensorflow' && document.getElementById('option-hddl-yes').classList.contains('selected')) {
        document.getElementById('option-hddl-yes').classList.remove('selected');
        document.getElementById('option-hddl-no').classList.add('selected');
    }
    if (div.id == 'option-hddl-yes' && document.getElementById('option-byo-tensorflow').classList.contains('selected')) {
        document.getElementById('option-byo-tensorflow').classList.remove('selected');
        document.getElementById('option-byo-none').classList.add('selected');
    }
    if (div.id == 'option-byo-openvino' && document.getElementById('option-hddl-no').classList.contains('selected') && document.getElementById('option-distro-pip').classList.contains('selected')) {
        document.getElementById('option-hddl-no').classList.remove('selected');
        document.getElementById('option-hddl-yes').classList.add('selected');
    }
    if (div.id == 'option-hddl-no' && document.getElementById('option-byo-openvino').classList.contains('selected') && document.getElementById('option-distro-pip').classList.contains('selected')) {
        document.getElementById('option-byo-openvino').classList.remove('selected');
        document.getElementById('option-byo-none').classList.add('selected');
    }
    if (div.id == 'option-distro-pip' && document.getElementById('option-byo-openvino').classList.contains('selected') && document.getElementById('option-hddl-no').classList.contains('selected')) {
        document.getElementById('option-hddl-yes').classList.add('selected');
        document.getElementById('option-hddl-no').classList.remove('selected');
    }
    if (div.id == 'option-distro-source' && document.getElementById('option-byo-tensorflow').classList.contains('selected') && document.getElementById('option-hddl-no').classList.contains('selected')) {
        document.getElementById('option-byo-none').classList.add('selected');
        document.getElementById('option-byo-tensorflow').classList.remove('selected');
    }
    if (div.id == 'option-byo-tensorflow' && document.getElementById('option-distro-source').classList.contains('selected') && document.getElementById('option-hddl-no').classList.contains('selected')) {
        document.getElementById('option-distro-pip').classList.add('selected');
        document.getElementById('option-distro-source').classList.remove('selected');
    }
    if (div.id == 'option-distro-source' && document.getElementById('option-byo-tensorflow').classList.contains('selected') && document.getElementById('option-hddl-no').classList.contains('selected')) {
        document.getElementById('option-distro-pip').classList.add('selected');
        document.getElementById('option-distro-source').classList.remove('selected');
    }
    if (div.id == 'option-hddl-no' && document.getElementById('option-byo-tensorflow').classList.contains('selected') && document.getElementById('option-distro-source').classList.contains('selected')) {
        document.getElementById('option-byo-none').classList.add('selected');
        document.getElementById('option-byo-tensorflow').classList.remove('selected');
    }
}

function removeSameRow(div) {
    const specs = div.id.split('-');
    const name = `${specs[0]}-${specs[1]}`;
    const othertags = document.querySelectorAll('*');
    for (let othertag of othertags) {
        if (othertag.id.startsWith(name) && othertag.id != div.id) {
            othertag.classList.remove('selected');
        }
    }
}

function commandArea() {
    const area = document.getElementById('command-area');

    const mapping = getByo() + getOs() + getDistro() + getHddl();
    const command = combination.get(mapping);
    console.log(mapping);
    if (!command) {
        let invalidConfig = document.createElement('b');
        invalidConfig.innerHTML = "";
        area.innerHTML = "";
        area.appendChild(invalidConfig);
    } else {
        area.innerHTML = "";
        for (let line of command.split(',')) {
            let paragraph = document.createElement('p');
            if (line.indexOf('Link') >= 0) {
                const name = document.createTextNode(line.substr(0, line.indexOf('Link')));
                let hyperlink = document.createElement('a');
                hyperlink.appendChild(name);
                console.log(line.substr(line.indexOf('Link') + 5))
                hyperlink.href = line.substr(line.indexOf('Link') + 5);
                paragraph.appendChild(hyperlink);
            } else {
                paragraph.innerHTML = line;
            }

            area.appendChild(paragraph);
        }
    }

}

function getByo() {
    if (document.getElementById('option-byo-none').classList.contains('selected')) return "00"; //return "00";
    if (document.getElementById('option-byo-tensorflow').classList.contains('selected')) return "01"; //return "01";
    if (document.getElementById('option-byo-openvino').classList.contains('selected')) return "10"; //return "10";
    // if(document.getElementById('option-byo-tfov').classList.contains('selected')) return "11"; //return "11";
    return "2";
}

function getOs() {
    if (document.getElementById('option-os-ubuntu18').classList.contains('selected')) return "00"; //return "00";
    if (document.getElementById('option-os-ubuntu20').classList.contains('selected')) return "01"; //return "01";
    if (document.getElementById('option-os-windows').classList.contains('selected')) return "10"; //return "10";
    if (document.getElementById('option-os-mac').classList.contains('selected')) return "11"; //return "11";
    return "2";
}

function getDistro() {
    if (document.getElementById('option-distro-pip').classList.contains('selected')) return "0"; //return "0";
    if (document.getElementById('option-distro-source').classList.contains('selected')) return "1"; //return "1";
    return "2";
}

function getHddl() {
    if (document.getElementById('option-hddl-yes').classList.contains('selected')) return "0"; //return "0";
    if (document.getElementById('option-hddl-no').classList.contains('selected')) return "1"; //return "1";
    return "2";
}
