import React from "react";
import "../style/home.css";
import photo from "../image/Okarun.jpg";
import download from "../image/download.png";
import coding from "../image/coding.png";
import email from "../image/email.png";
import github from "../image/github.png";
import instagram from "../image/instagram.png";
import linkedin from "../image/linkedin.png";
import Conta from "./Conta";
import resume from "../image/resume.png";
const Home = () => {
  return (
    <div className="mains">
      <div className="home_main">
        <div id="name">
          <h1>Hello,I'M</h1>
          <h1>GOTHANDARAMAN</h1>
          <p>Frontend Developer | And More....</p>
          <div id="resume">
            <button>
              <a href={resume} target="_black">
                Get Resume
                <img src={download} alt="" />
              </a>
            </button>
          </div>
        </div>
        <div className="photo">
          <img src={photo} alt="" />

          <div className="icon">
            <img src={coding} alt="" />

            <img src={github} alt="" />

            <img src={email} alt="" />

            <img src={linkedin} alt="" />

            <img src={instagram} alt="" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
