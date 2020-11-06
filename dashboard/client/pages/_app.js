import React from "react";
import App from "next/app";

import stylesheet from 'antd/dist/antd.min.css'
import '../styles/globals.css';

class MyApp extends App {
  render() {
    const { Component, pageProps } = this.props;

    return <Component {...pageProps} />;
  }
}

export default MyApp;
