
const combination = new Map([
    ["000001","pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tensorflow-security-patched-abi0,pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ openvino-tensorflow-addon-abi0"],
    ["010001","# Assumes user desired TF is installed through pip,pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ openvino-tensorflow-addon-abi0"],
    ["000011","# This will install recommended security patched TF from PyPi,python3 build_ovtf.py --use_prebuilt_tensorflow"],
    ["010011","# Coming Soon: Assumes user desired TF is installed through pip,python3 build_ovtf.py --use_prebuilt_tensorflow --use_tensorflow_installation"],
    ["000000","pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tensorflow-security-patched-abi1,pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ openvino-tensorflow-addon-abi1"],
    ["100000","# Assumes OpenVINO installed through following methods,# pip install openvino (or) source /opt/intel/openvino/bin/setupvars.sh,pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tensorflow-security-patched-abi1,pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ openvino-tensorflow-addon-abi1"],
    ["000010","# This will install recommended security patched TF from PyPi,python3 build_ovtf.py --use_prebuilt_tensorflow --cxx11_abi_version=1"],
    ["100010","# Coming Soon: This will use OpenVINO binary installation under /opt/intel/openvino/,python3 build_ovtf.py --use_prebuilt_tensorflow --cxx11_abi_version=1 --use_openvino_installation=/opt/intel/openvino"]
]);

window.onload =new function (){
    assignfunction();
    
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
                removeOther(div);
            }
        }
    }
}
function removeOther(div)
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
    commandArea();     
}
function commandArea()
{
    const area = document.getElementById('command-area');
    const mapping = getByo()+getOs()+getDistro()+getHddl();
    const command = combination.get(mapping);
    console.log(mapping);
    // if(!command)
    // {
    //     area.innerHTML = "Not Supported / Choose Features";
    // }
    // else
    // {
    //     area.innerHTML = "";
    //     for(let line of command.split(','))
    //     {
    //         let paragraph = document.createElement('p');
    //         paragraph.innerHTML = line;
    //         area.appendChild(paragraph);
    //     }
    // }

    if(mapping.indexOf('2')==-1)
    {
        area.innerHTML = `Installed: ${getByo()}  OS: ${getOs()} Distro: ${getDistro()} HDDL: ${getHddl()}`;
    }
    else
    {
        area.innerHTML = "Choose Features";
    }
    
}
function getByo()
{
    if(document.getElementById('option-byo-none').classList.contains('selected')) return "none"; //return "00";
    if(document.getElementById('option-byo-tensorflow').classList.contains('selected')) return "tensorflow"; //return "01";
    if(document.getElementById('option-byo-openvino').classList.contains('selected')) return "openvino"; //return "10";
    if(document.getElementById('option-byo-tfov').classList.contains('selected')) return "openvino&tensorflow"; //return "11";
    return "2";
}

function getOs()
{
    if(document.getElementById('option-os-ubuntu').classList.contains('selected')) return "ubuntu"; //return "00";
    if(document.getElementById('option-os-windows').classList.contains('selected')) return "windows";//return "01";
    if(document.getElementById('option-os-mac').classList.contains('selected')) return "Mac OS";//return "10";
    return "2";
}

function getDistro()
{
    if(document.getElementById('option-distro-pip').classList.contains('selected')) return "pip";//return "0";
    if(document.getElementById('option-distro-source').classList.contains('selected')) return "source";//return "1";
    return "2";
}

function getHddl()
{
    if(document.getElementById('option-hddl-yes').classList.contains('selected')) return "yes";//return "0";
    if(document.getElementById('option-hddl-no').classList.contains('selected')) return "no";//return "1";
    return "2";
}