import React from "react";
import "../style/contact.css";
import github from "../image/github.png";
import instagram from "../image/instagram.png";
import linkedin from "../image/linkedin.png";
import { Link } from "react-router-dom";
import Conta from "./Conta";
const Contact = () => {
  function github1() {
    location.href = "https://github.com/yerfor/GeneFace/blob/main/data/";
  }
  function instagram1() {
    location.href =
      "https://www.instagram.com/logic_error_76?igsh=MXZweDJ1NW42OWRyOQ==";
  }
  function linkedin1() {
    location.href = "https://www.linkedin.com/in/gothanda-raman-261513274";
  }
  return (
    <div className="main_contact">
      <div className="about_box">
        <h1>Contact</h1>
        <p>
          <Link to="/home">Home</Link> || <a href="#">Let's Contact</a>
        </p>
      </div>
      <div className="contact_det">
        <div id="two">
          <div id="email">
            <img src="email.png" alt="" />
            <span>Email</span>
            <a href="mailto:gothandaraman314@gamil.com">
              {" "}
              gothandaraman314@gamil.com
            </a>
          </div>
          <div id="email">
            {/* <img src="iphone.png" alt="Phone Icon" /> */}
            <span>Phone</span>
            <a href="tel:+916374464023">+91 6374464023</a>
          </div>
          <div id="email">
            <img src="linkedin.png" alt="" />
            <span>Linked In</span>
            <a href="https://www.linkedin.com/in/gothanda-raman-261513274">
              @Gothan
            </a>
          </div>
        </div>
        <h2>Other Social Links</h2>
        <div id="icon1">
          <img onClick={github1} src={github} alt="" />
          <img onClick={instagram1} src={instagram} alt="" />
          <img onClick={linkedin1} src={linkedin} alt="" />
        </div>
      </div>
    </div>
  );
};

export default Contact;
