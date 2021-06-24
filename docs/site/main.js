
const combination = new Map([
    ["000000","# For Python3.7 and Python3.8 change the versions in the links appropriately , pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.5.0-cp36-cp36m-manylinux2010_x86_64.whl , pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp36-cp36m-manylinux2014_x86_64.whl , Download Intel® Distribution of OpenVINO™ Toolkit Link-https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?operatingsystem=linux&distributions=webdownload&version=2021%203%20(latest)&options=offline , source $INTEL_OPENVINO_DIR/bin/setupvars.sh"
],
["000100"," pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp38-cp38-linux_x86_64.whl , pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.5.0-cp38-cp38-manylinux2010_x86_64.whl , Download Intel® Distribution of OpenVINO™ Toolkit Link-https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?operatingsystem=linux&distributions=webdownload&version=2021%203%20(latest)&options=offline , source $INTEL_OPENVINO_DIR/bin/setupvars.sh"
],
["000001"," pip3 install -U pip==21.0.1,pip3 install -U tensorflow==2.5.0,pip3 install openvino-tensorflow"
],
["000101"," pip3 install -U pip==21.0.1,pip3 install -U tensorflow==2.5.0,pip3 install openvino-tensorflow"
],
["010001","pip3 install openvino-tensorflow"
],
["010101","pip3 install openvino-tensorflow"
],
["100000","# For Python3.7 and Python3.8 change the versions in the links appropriately , pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.5.0-cp36-cp36m-manylinux2010_x86_64.whl , pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp36-cp36m-manylinux2014_x86_64.whl , source $INTEL_OPENVINO_DIR/bin/setupvars.sh"
],
["100100"," pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp38-cp38-linux_x86_64.whl , pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.5.0-cp38-cp38-manylinux2010_x86_64.whl , source $INTEL_OPENVINO_DIR/bin/setupvars.sh"
],
["000010"," Download Intel® Distribution of OpenVINO™ Toolkit Link-https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?operatingsystem=linux&distributions=webdownload&version=2021%203%20(latest)&options=offline ,python3 build_ovtf.py --use_openvino_from_location=$INTEL_OPENVINO_DIR --cxx11_abi_version=1"
],
["000110"," Download Intel® Distribution of OpenVINO™ Toolkit Link-https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?operatingsystem=linux&distributions=webdownload&version=2021%203%20(latest)&options=offline ,python3 build_ovtf.py --use_openvino_from_location=$INTEL_OPENVINO_DIR --cxx11_abi_version=1"
],
["000011","python3 build_ovtf.py "
],
["000111","python3 build_ovtf.py "
],
["100010","#Use OpenVINO 2021.3,python3 build_ovtf.py --use_openvino_from_location=$INTEL_OPENVINO_DIR --cxx11_abi_version=1"
],
["100110","#Use OpenVINO 2021.3,python3 build_ovtf.py --use_openvino_from_location=$INTEL_OPENVINO_DIR --cxx11_abi_version=1"
],
["100011","python3 build_ovtf.py"
],
["100111","python3 build_ovtf.py"
]
    ]);

window.onload =new function (){
    assignfunction();
    commandArea();
    
};

function assignfunction()
{
    const divs = document.querySelectorAll('div');

    for(let div of divs)
    {
        if(div.id.startsWith('option'))
        {
            div.onclick = ()=>{
                if(div.classList.contains('selected'))
                {
                    div.classList.remove('selected');
                }
                else{
                    div.classList.add('selected');
                }
                removeSameRow(div);
                removeFromOtherRow(div);
                commandArea(); 
            }
        }
    }
}
function removeFromOtherRow(div)
{
    if(div.id=='option-byo-tensorflow' && document.getElementById('option-hddl-yes').classList.contains('selected'))
    {
        document.getElementById('option-hddl-yes').classList.remove('selected');
        document.getElementById('option-hddl-no').classList.add('selected');
    }
    if(div.id=='option-hddl-yes' && document.getElementById('option-byo-tensorflow').classList.contains('selected'))
    {
        document.getElementById('option-byo-tensorflow').classList.remove('selected');
        document.getElementById('option-byo-none').classList.add('selected');
    }
    if(div.id=='option-byo-openvino' && document.getElementById('option-hddl-no').classList.contains('selected')&&document.getElementById('option-distro-pip').classList.contains('selected'))
    {
        document.getElementById('option-hddl-no').classList.remove('selected');
        document.getElementById('option-hddl-yes').classList.add('selected');
    }
    if(div.id=='option-hddl-no' && document.getElementById('option-byo-openvino').classList.contains('selected')&&document.getElementById('option-distro-pip').classList.contains('selected'))
    {
        document.getElementById('option-byo-openvino').classList.remove('selected');
        document.getElementById('option-byo-none').classList.add('selected');
    }
    if(div.id=='option-distro-pip' && document.getElementById('option-byo-openvino').classList.contains('selected')&&document.getElementById('option-hddl-no').classList.contains('selected'))
    {
        document.getElementById('option-hddl-yes').classList.add('selected');
        document.getElementById('option-hddl-no').classList.remove('selected');
    }
    if(div.id=='option-distro-source' && document.getElementById('option-byo-tensorflow').classList.contains('selected')&&document.getElementById('option-hddl-no').classList.contains('selected'))
    {
        document.getElementById('option-byo-none').classList.add('selected');
        document.getElementById('option-byo-tensorflow').classList.remove('selected');
    }
    if(div.id=='option-byo-tensorflow' && document.getElementById('option-distro-source').classList.contains('selected')&&document.getElementById('option-hddl-no').classList.contains('selected'))
    {
        document.getElementById('option-distro-pip').classList.add('selected');
        document.getElementById('option-distro-source').classList.remove('selected');
    }
    if(div.id=='option-distro-source' && document.getElementById('option-byo-tensorflow').classList.contains('selected')&&document.getElementById('option-hddl-no').classList.contains('selected'))
    {
        document.getElementById('option-distro-pip').classList.add('selected');
        document.getElementById('option-distro-source').classList.remove('selected');
    }
    if(div.id=='option-hddl-no' && document.getElementById('option-byo-tensorflow').classList.contains('selected')&&document.getElementById('option-distro-source').classList.contains('selected'))
    {
        document.getElementById('option-byo-none').classList.add('selected');
        document.getElementById('option-byo-tensorflow').classList.remove('selected');
    }
}
function removeSameRow(div)
{
    const specs= div.id.split('-');
    const name = `${specs[0]}-${specs[1]}`;
    const othertags = document.querySelectorAll('*');
    for(let othertag of othertags)
    {
        if(othertag.id.startsWith(name) && othertag.id!=div.id)
        {
            othertag.classList.remove('selected');
        }
    }    
}
function commandArea()
{
    const area = document.getElementById('command-area');
    // For OS Temporary Output
    if(getOs()=='10'||getOs()=='11')
    {
        let nameOS = getOs()=='10'?"Windows":"MacOS";
        area.innerHTML = `${nameOS} installation not supported`;
        return;
    }
    
    const mapping = getByo()+getOs()+getDistro()+getHddl();
    const command = combination.get(mapping);
    console.log(mapping);
    if(!command)
    {
      let invalidConfig = document.createElement('b');
      invalidConfig.innerHTML = "";
      area.innerHTML = "";
      area.appendChild(invalidConfig);
    }
    else
    {
        area.innerHTML = "";
        for(let line of command.split(','))
        {
            let paragraph = document.createElement('p');
            if(line.indexOf('Link')>=0)
            {
                const name = document.createTextNode(line.substr(0,line.indexOf('Link')));
                let hyperlink = document.createElement('a');
                hyperlink.appendChild(name);
                console.log(line.substr(line.indexOf('Link')+5))
                hyperlink.href = line.substr(line.indexOf('Link')+5);
                paragraph.appendChild(hyperlink);
            }
            else
            {
                paragraph.innerHTML = line;
            }
            
            area.appendChild(paragraph);
        }
    }
    
}
function getByo()
{
    if(document.getElementById('option-byo-none').classList.contains('selected')) return "00"; //return "00";
    if(document.getElementById('option-byo-tensorflow').classList.contains('selected')) return "01"; //return "01";
    if(document.getElementById('option-byo-openvino').classList.contains('selected')) return "10"; //return "10";
    // if(document.getElementById('option-byo-tfov').classList.contains('selected')) return "11"; //return "11";
    return "2";
}

function getOs()
{
    if(document.getElementById('option-os-ubuntu18').classList.contains('selected')) return "00"; //return "00";
    if(document.getElementById('option-os-ubuntu20').classList.contains('selected')) return "01"; //return "01";
    if(document.getElementById('option-os-windows').classList.contains('selected')) return "10";//return "10";
    if(document.getElementById('option-os-mac').classList.contains('selected')) return "11";//return "11";
    return "2";
}

function getDistro()
{
    if(document.getElementById('option-distro-pip').classList.contains('selected')) return "0";//return "0";
    if(document.getElementById('option-distro-source').classList.contains('selected')) return "1";//return "1";
    return "2";
}

function getHddl()
{
    if(document.getElementById('option-hddl-yes').classList.contains('selected')) return "0";//return "0";
    if(document.getElementById('option-hddl-no').classList.contains('selected')) return "1";//return "1";
    return "2";
}