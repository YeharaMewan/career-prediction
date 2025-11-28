import React from "react";
import Button from "../../components/Button";
import Graph from "../../components/Graph";
import Border from "../../components/border";

const careerPath: React.FC = () => {
  return (

<>
<div className="px-8 py-8 mt-5">
  <h1 className="text-4xl font-bold font-mono text-gray-800 uppercase">
    Explore career paths
  </h1>
    <p className="mt-2 text-gray-600 font-sans">Powered by AI-Driven Matching & Analytics</p>
</div>
<div className="p-10 flex flex-col  items-center justify-center ">
    <div className="grid  grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-16 w-full">
        <Border borderRadius="1rem">
        <div className="flex flex-row ">
            <div className="w-2/3">
            <h2 className="font-mono text-black uppercase font-semibold text-xl">ai reasearch scientist</h2>
            <div className="flex flex-col mt-6">
                <p className="font-sans">Pioneering new algorithms and  machine learning modes</p>
                <div className="mt-4 mb-4">
                    
                    <ul className="list-disc list-disc-blue list-inside mt-2 font-sans">
                        <li>Advanced Maths Skills</li>
                        <li>Problem Solving Aptitude</li>
                        <li>Coding Proficiency</li>
                
                    </ul>
                    <div className="my-3">
                    <Button label="Select this path" className=" uppercase font-mono bg-blue-400"/>

                    </div>
                    </div>
            </div>
            </div>
            <div className="p-2 flex justify-center items-center">
              <Graph value={92} label="MATCH" />
            </div>
        </div>
        </Border>
                <Border borderRadius="1rem">
                <div className="flex flex-row">
            <div className="w-2/3">
            <h2 className="font-mono text-black uppercase font-semibold text-xl">quantuitative analyst</h2>
            <div className="flex flex-col mt-6">
                <p className="font-sans 
                ">Developing financial models and trading startegies.</p>
                <div className="mt-4 mb-4">
                    
                    <ul className="list-disc list-disc-blue list-inside mt-2 font-sans">
                        <li>Strong Analytical Mind</li>
                        <li>Date Driven Decisions</li>
                        <li>Programming (Python / R)</li>
                
                    </ul>
                    <div className="my-3">
                    <Button label="Select this path" className=" uppercase font-mono bg-green-400"/>

                    </div>
                    </div>
            </div>
            </div>
            <div className="p-2 flex justify-center items-center">
              <Graph value={88} label="MATCH" strokeColor="green" />
            </div>
        </div>
        </Border>

        <Border borderRadius="1rem">
          <div className="flex flex-row  rounded-2xl">
            <div className="w-2/3">
            <h2 className="font-mono text-black uppercase font-semibold text-xl">cyber security specialist</h2>
            <div className="flex flex-col mt-6">
                <p className="font-sans 
                ">Safeguarding critical digital systems by preventing, detecting, and responding to modern cyber threats.</p>
                <div className="mt-4 mb-4">
                    
                    <ul className="list-disc list-disc-blue list-inside mt-2 font-sans">
                        <li>Strong Analytical Mind</li>
                        <li>Date Driven Decisions</li>
                        <li>Programming (Python / R)</li>
                
                    </ul>
                    <div className="my-3">
                    <Button label="Select this path" className=" uppercase font-mono bg-orange-400"/>

                    </div>
                    </div>
            </div>
            </div>
            <div className="p-2 flex justify-center items-center">
              <Graph value={75} label="MATCH" strokeColor="orange" />
            </div>
        </div>
        </Border>




        <Border borderRadius="1rem">
          <div className="flex flex-row  rounded-2xl">
            <div className="w-2/3">
            <h2 className="font-mono text-black uppercase font-semibold text-xl">robotics engineer</h2>
            <div className="flex flex-col mt-6">
                <p className="font-sans 
                ">Engineering advanced robotic technologies that enhance automation and human efficiency</p>
                <div className="mt-4 mb-4">
                    
                    <ul className="list-disc list-disc-blue list-inside mt-2 font-sans">
                        <li>Strong Analytical Mind</li>
                        <li>Date Driven Decisions</li>
                        <li>Programming (Python / R)</li>
                
                    </ul>
                    <div className="my-3">
                    <Button label="Select this path" className=" uppercase font-mono bg-purple-400"/>

                    </div>
                    </div>
            </div>
            </div>
            <div className="p-2 flex justify-center items-center">
              <Graph value={60} label="MATCH" strokeColor="purple" />
            </div>
        </div>
        </Border>
    </div>
</div>
</>

  );
};

export default careerPath;