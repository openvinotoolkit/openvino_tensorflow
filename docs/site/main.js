
const combination = new Map([
    ["000000","pip install https://github.com/openvinotoolkit/openvino_tensorflow/releases/tensorflow-abi1.whl , pip install https://github.com/openvinotoolkit/openvino_tensorflow/releases/openvino-tensorflow-abi1.whl , Download Intel® Distribution of OpenVINO™ Toolkit Link-https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?operatingsystem=linux&distributions=webdownload&version=2021%203%20(latest)&options=offline , source /opt/intel/openvino_2021.3.394/bin/setupvars.sh"
],
["000001","pip install tensorflow , pip install openvino-tensorflow"
],
["010001","pip install openvino-tensorflow"
],
["100000","pip install https://github.com/openvinotoolkit/openvino_tensorflow/releases/tensorflow-abi1.whl , pip install https://github.com/openvinotoolkit/openvino_tensorflow/releases/openvino-tensorflow-abi1.whl , source /opt/intel/openvino_2021.3.394/bin/setupvars.sh"
],
["000010"," Download Intel® Distribution of OpenVINO™ Toolkit Link-https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?operatingsystem=linux&distributions=webdownload&version=2021%203%20(latest)&options=offline ,python3 build_ovtf.py --use_openvino_from_location=/opt/intel/openvino_2021.3.394/"
],
["000011","python3 build_ovtf.py"
],
["100010","#Use OpenVINO 2021.3,python3 build_ovtf.py --use_openvino_from_location=/opt/intel/openvino_2021.3.394/"
],
["100011","python3 build_ovtf.py"
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
    // Blurr Effect when openvino/tensorflow is selected
    // if(document.getElementById('option-byo-tfov').classList.contains('selected'))
    // {
    //     document.getElementById('option-hddl-no').classList.add('blur');
    //     document.getElementById('option-hddl-yes').classList.add('blur');
    // }
    // else
    // {
    //     document.getElementById('option-hddl-no').classList.remove('blur');
    //     document.getElementById('option-hddl-yes').classList.remove('blur');
    // }

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
    if(getOs()=='01'||getOs()=='10')
    {
        let nameOS = getOs()=='01'?"Windows":"MAC OS";
        area.innerHTML = `${nameOS} installation not supported`;
        return;
    }
    
    const mapping = getByo()+getOs()+getDistro()+getHddl();
    const command = combination.get(mapping);
    console.log(mapping);
    if(!command)
    {
        area.innerHTML = "Choose Features/Invalid Option";
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
    if(document.getElementById('option-os-ubuntu').classList.contains('selected')) return "00"; //return "00";
    if(document.getElementById('option-os-windows').classList.contains('selected')) return "01";//return "01";
    if(document.getElementById('option-os-mac').classList.contains('selected')) return "10";//return "10";
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