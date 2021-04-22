import { Layout, Menu, Breadcrumb } from 'antd';
import { UserOutlined, LaptopOutlined, NotificationOutlined } from '@ant-design/icons';
import { Input, Result } from 'antd';
import { Spin, Space } from 'antd';
import { DatabaseOutlined } from '@ant-design/icons';

import { useState, useEffect } from 'react';
import { observer } from "mobx-react"

import { useRouter } from 'next/router'

import dm from '../models/DataManager';

const { SubMenu } = Menu;
const { Header, Content, Footer, Sider } = Layout;
const { Search } = Input;

import DetailView from '../components/DetailView';

const suffix = (
  <DatabaseOutlined
    style={{
      fontSize: 16,
      color: '#1890ff',
    }}
  />
);

const ServerSelector = ({ setServer,  loading, startValue }) => {

  const [curValue, setValue] = useState('')

  useEffect(x => {
    setValue(startValue)
  }, [ startValue ]);

  return <>
    <Search
      style={{ marginTop: '12px'}}
      enterButton="Connect"
      value={curValue}
      size="large"
      loading={loading}
      suffix={suffix}
      prefix="http://"
      onChange={e => setValue(e.target.value)}
      onSearch={x => setServer(x)}
    />
  </>
}


export default observer(() => {
  const router = useRouter()
  const [currentServer, setServer ] = useState('');

  if (currentServer === '' && typeof(router.query.url) !== 'undefined') {
    setServer(router.query.url);
  }

  console.log(currentServer);
  if(currentServer != '') {
    dm.fetchUrl('http://' + currentServer);
  }

  const loading = !dm.loaded && currentServer != '' && !dm.failed;
  
  let main_message = <div>Fill the url of your 3DB dashboard API</div>;
  let menu = null;
  
  if(dm.failed) {
    main_message = (
    <Result
      status="500"
      title="Error loading the data"
      subTitle="Are you sure the Url is correct and the API server is running ?"
    />);
  }
  else if(dm.loaded) {
    menu = (
      <Menu style={{ display:'inline-block' }}theme="dark" mode="horizontal" defaultSelectedKeys={['details']}>
        <Menu.Item key="home">Summary</Menu.Item>
        <Menu.Item key="details">Detail view</Menu.Item>
        <Menu.Item key="stratified">Analytics</Menu.Item>
      </Menu>
    );
    main_message = <DetailView />;
  }
  else if (loading) {
    main_message = <Spin size="large" style={{ margin: 'auto' }}/>;
  }

  return <>
    <Layout style={{ height: '100%' }}>
      <Header className="header">
        <div style={{display: 'inline-block', marginRight: '15px', color: 'white', fontWeight:'bold'}}>3DB</div>
        {menu}
        <div style={{position: 'absolute', right: '12px', top:0, color: 'white'}}>
          <ServerSelector setServer={setServer} loading={loading} startValue={currentServer}/>
        </div>
      </Header>
      <Content style={{ padding: '0 50px', height: '100%' }}>
        <Layout className="site-layout-background" style={{ padding: '24px 0', height: '100%' }}>
          <Content style={{ padding: '0 24px', minHeight: 280, height:'100%' }}>
            {main_message}
          </Content>
        </Layout>
      </Content>
        <Footer style={{ textAlign: 'center' }}>Madry Lab ©2020 </Footer>
    </Layout>
  </>
})
