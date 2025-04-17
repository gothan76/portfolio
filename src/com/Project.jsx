import React from "react";
import "../style/project.css";
import qr from '../image/qr.png'
import weather from "../image/weather.png"
import amazon from '../image/amazon.png'
import live from "../image/live.png";
import css from "../image/css.png";
import html from "../image/html.png";
import js from "../image/js.png";
import react from "../image/react.png";
import github from "../image/github.png";
import Conta from "./Conta";
import { Link } from "react-router-dom";

 function github1() {
    location.href = "https://github.com/Rishi496/QR-Generator";
  }
   function github2() {
     location.href = "https://github.com/Rishi496/Weather";
   }
    function github3() {
      location.href = "https://github.com/Rishi496/amazon-clone";
    }
    function live1() {
      location.href = "https://qr-generator-pink-phi.vercel.app/";
    }
     function live2() {
       location.href = "https://weather-mu-six.vercel.app/";
     }
    function live3() {
      location.href = "https://amazon-clone-pied-chi.vercel.app/";
    }
const Project = () => {
  return (
    <div className="main_project">
      <div className="about_box">
        <h1>Projects</h1>
        <p>
          <Link to="/home">Home</Link> || <a href="#">Projects List</a>
        </p>
      </div>
      <div className="project_det">
        <div className="pro_det">
          <h1>QR Generator</h1>
          <p>
            Create a system that listens to conversations between doctors and
            patients, turns the spoken words into text, and organizes them into
            a clear, structured medical report
          </p>
          <img src={qr} alt="" />
          <div className="box">
            <div className="first">
              <span>
                <img src={html} alt="" />
                <img src={css} alt="" />
                <img src={js} alt="" />
                <img src={react} alt="" />
              </span>
              {/* <button onClick={github1}>
                <img src={github} alt="" />
                Start the project
              </button> */}
            </div>
            <div className="second">
              <button onClick={live1}>
                <img src={live} alt="" />
              </button>
            </div>
          </div>
        </div>
        <div className="pro_det">
          <h1>Weather</h1>
          <p>
            Create a system that listens to conversations between doctors and
            patients, turns the spoken words into text, and organizes them into
            a clear, structured medical report
          </p>
          <img src={weather} alt="" />
          <div className="box">
            <div className="first">
              <span>
                <img src={html} alt="" />
                <img src={css} alt="" />
                <img src={js} alt="" />
                <img src={react} alt="" />
              </span>
              {/* <button onClick={github2}>
                <img src={github} alt="" />
                Start the project
              </button> */}
            </div>
            <div className="second">
              <button onClick={live2}>
                <img src={live} alt="" />
              </button>
            </div>
          </div>
        </div>
        <div className="pro_det">
          <h1>Amazon Clone</h1>
          <p>
            Create a system that listens to conversations between doctors and
            patients, turns the spoken words into text, and organizes them into
            a clear, structured medical report
          </p>
          <img src={amazon} alt="" />
          <div className="box">
            <div className="first">
              <span>
                <img src={html} alt="" />
                <img src={css} alt="" />
                <img src={js} alt="" />
                <img src={react} alt="" />
              </span>
              {/* <button onClick={github3}>
                <img src={github} alt="" />
                Start the project
              </button> */}
            </div>
            <div className="second">
              <button onClick={live3}>
                <img src={live} alt="" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Project;
