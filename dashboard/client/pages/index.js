import { Layout, Menu, Breadcrumb } from 'antd';
import { UserOutlined, LaptopOutlined, NotificationOutlined } from '@ant-design/icons';
import { Input } from 'antd';
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
      onChange={e => setValue(e.target.value)}
      onSearch={setServer}
    />
  </>
}


export default observer(() => {
  const router = useRouter()
  const [currentServer, setServer ] = useState('');

  if (currentServer === '' && typeof(router.query.url) !== 'undefined') {
    setServer(router.query.url);
  }

  if(currentServer != '') {
    dm.fetchUrl(currentServer);
  }

  const footer = <Footer style={{ textAlign: 'center' }}>Madry Lab ©2020 </Footer>

  if (!dm.loaded) {
    return <>
      <Layout style={{ height: '100%' }}>
        <Header className="header">
          <div style={{display: 'inline-block', marginRight: '15px', color: 'white', fontWeight:'bold'}}>Synthetic-Sandbox</div>
          <div style={{position: 'absolute', right: '12px', top:0, color: 'white'}}>
            <ServerSelector setServer={setServer} loading={!dm.loaded} startValue={currentServer}/>
          </div>
        </Header>
        <Content style={{ padding: '0 50px', height: '100%' }}>
          <Layout className="site-layout-background" style={{ padding: '24px 0', height: '100%' }}>
            <Content style={{ padding: '0 24px', minHeight: 280, textAlign: 'center', height:'100%' }}>
              <Spin size="large" style={{ margin: 'auto' }}/>
            </Content>
          </Layout>
        </Content>
          { footer }
      </Layout>
    </>
  }

  return <>
    <Layout>
      <Header className="header">
        <div style={{display: 'inline-block', marginRight: '15px', color: 'white', fontWeight:'bold'}}>Synthetic-Sandbox</div>
        <Menu style={{ display:'inline-block' }}theme="dark" mode="horizontal" defaultSelectedKeys={['details']}>
          <Menu.Item key="home">Summary</Menu.Item>
          <Menu.Item key="details">Detail view</Menu.Item>
          <Menu.Item key="stratified">Analytics</Menu.Item>
        </Menu>
        <div style={{position: 'absolute', right: '12px', top:0, color: 'white'}}>
          <ServerSelector setServer={setServer} loading={!dm.loaded} startValue={currentServer}/>
        </div>
      </Header>
      <Content style={{ padding: '0 50px' }}>
        <Layout className="site-layout-background" style={{ padding: '24px 0' }}>
          <Content style={{ padding: '0 24px', minHeight: 280 }}>
            <DetailView />
          </Content>
        </Layout>
      </Content>
        { footer }
    </Layout>
  </>
})
