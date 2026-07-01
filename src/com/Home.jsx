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
            {/* <a
              className="navimage"
              href="https://github.com/gothan76/Portfolio/tree/main/src/com"
            >
              <img src={coding} alt="" />{" "}
            </a> */}

            <a
              href="https://github.com/gothan76"
              target="_blank"
            >
              {" "}
              <img src={github} alt="" />
            </a>
            <a
              href="mailto:gothandaraman314@gmail.com"
              target="_blank"
            >
              <img src={email} alt="" />
            </a>
            <a
              href="https://www.linkedin.com/in/gothanda-raman-261513274/"
              target="_blank"
            >
              <img src={linkedin} alt="" />
            </a>
            <a href="">
              <img src={instagram} alt="" />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
