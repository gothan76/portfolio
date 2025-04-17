import React, { useState } from "react";
import download from "../image/download.png";
import "../style/about.css";
import Conta from "./Conta";
import resume from "../image/resume.png";
import { Link } from "react-router-dom";

const About = () => {
  const [show, setShow] = useState("education");
  return (
    <div className="main_about">
      <div className="about_box">
        <h1>About Us</h1>
        <p>
          <Link to="/home">Home</Link> || <a href="#">About</a>
        </p>
      </div>
      {/* --------------------------------------------------------- */}
      <div className="my_detial">
        <div className="resume">
          <button>
            <a href={resume} target="_black">
              Get Resume
              <img src={download} alt="" />
            </a>
          </button>
        </div>
        <div className="name">
          <h1>I'M GOTHANDARAMAN</h1>
          <p className="para">Frontend Developer | And More....</p>
          <p>
            My name is GOTHANDARAMAN, and I am currently in the final year of my
            Bachelor of Engineering in Information Technology. I have a strong
            passion for technology and problem-solving, with a particular
            interest in becoming a full Frontend develope and I am eager to
            apply this knowledge to create innovative and efficient solutions. I
            am excited to continue growing as a developer and contribute to
            projects that make a real-world impact.
          </p>
        </div>
      </div>
      {/* --------------------------------------------------------- */}
      <div className="education1">
        <div className="but">
          <button
            onClick={(e) => {
              e.preventDefault();
              console.log(show);
              setShow("education");
            }}
          >
            Education
          </button>
          <button
            onClick={(e) => {
              e.preventDefault();
              console.log(show);
              setShow("experience");
            }}
          >
            Experience
          </button>
        </div>
        <div
          className={`${show === "education" ? "show" : "donotshow"}`}
          style={{}}
        >
          <div className="educa-det">
            <h1>ANNAMALAI UNIVERSITY:</h1>
            <p>BE Information Technology</p>
            <p>OGPA-8.12 (upto 6th sem)</p>
            <p>2021-2025</p>
          </div>
        </div>
        <div className={`${show === "experience" ? "show" : "donotshow"}`}>
          <div className="educa-det">
            <p className="year">25 March 2025 - Now</p>
            <h1>Murven Designs and Solutions</h1>
            <p>
              I am currently internning as a Frontend developer at Murven Design
              and Solutions , where i work on building responsive and user
              friendly website using React js.I collabrate with the team to turn
              ideas into real webpage.This experience has improved my coding
              skills and given me exposure to real world development practices.
            </p>
            <p className="year">1June 2025 -1 July 2025</p>
            <h1>Web Development Internship at CODSOFT</h1>
            <p>
              During my internship period , I developed my first portfolio
              website,Calulator and Landing page.
            </p>
            <p>Languages used:HTML,CSS,JAVASCRIPT.</p>
          </div>
        </div>
      </div>{" "}
      <div className="education">
        <div className="but">
          <button
            onClick={(e) => {
              e.preventDefault();
              console.log(show);
              setShow("education");
            }}
          >
            Education
          </button>
          <button
            onClick={(e) => {
              e.preventDefault();
              console.log(show);
              setShow("experience");
            }}
          >
            Experience
          </button>
        </div>
        <div
          className={`${show === "education" ? "show" : "donotshow"}`}
          style={{}}
        >
          <div className="year">
            {/* <p>2009</p>
            <p>2010</p>
            <p>2011</p> */}
          </div>
          <div className="educa-det">
            <h1>ANNAMALAI UNIVERSITY:</h1>
            <p>BE Information Technology</p>
            <p>OGPA-8.12 (upto 6th sem)</p>
            <p>2021-2025</p>
          </div>
        </div>
        <div className={`${show === "experience" ? "show" : "donotshow"}`}>
          <div className="year">
            <p>2009</p>
            <p>2010</p>
            <p>2011</p>
          </div>
          <div className="educa-det">
            <p style={{color:"blue"}}>25 March 2025 - Now</p>
            <h1>Murven Designs and Solutions</h1>
            <p>
              I am currently internning as a Frontend developer at Murven Design
              and Solutions , where i work on building responsive and user
              friendly website using React js.I collabrate with the team to turn
              ideas into real webpage.This experience has improved my coding
              skills and given me exposure to real world development practices.
            </p>
            <p style={{color:"blue"}}>1June 2025 -1 July 2025</p>
            <h1>Web Development Internship at CODSOFT</h1>
            <p>
              During my internship period , I developed my first portfolio
              website,Calulator and Landing page.
            </p>
            <p>Languages used:HTML,CSS,JAVASCRIPT.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
