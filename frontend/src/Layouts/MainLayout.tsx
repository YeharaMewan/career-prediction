import LiquidEther from "../components/LiquidEther";
//import Navbar from "../components/Navbar";

interface LayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="relative mx-auto min-h-screen w-full max-w-[1920px] overflow-y-hidden bg-[color:var(--color-main)]">
      <div className="fixed inset-0 z-10 h-full w-full">
        <LiquidEther
          colors={["#5227FF", "#FF9FFC", "#B19EEF"]}
          // colors={["#FF3300", "#FF9900", "#FFD633"]}
          //colors={["#FF2200", "#FF5500", "#FFAA22"]}

          //colors= {["#0044FF", "#00FFAA", "#1AC6FF"]}
          mouseForce={20}
          cursorSize={100}
          isViscous={false}
          viscous={30}
          iterationsViscous={32}
          iterationsPoisson={32}
          resolution={0.5}
          isBounce={false}
          autoDemo={true}
          autoSpeed={0.5}
          autoIntensity={2.2}
          takeoverDuration={0.25}
          autoResumeDelay={3000}
          autoRampDuration={0.6}
          style={{ width: "100%", height: "100%" }} 
        />
      </div>
      {/* <Navbar /> */}
      <main className="relative z-[10] px-[50px] py-[50px] xl:px-[100px]">{children}</main>
    </div>
  );
};

export default MainLayout;
