use std::time::Duration;

pub trait ReqwestClientTrait
{
    fn client_mut(&mut self) -> &mut ReqwestClient;
    fn client(&self) -> &ReqwestClient;
}

pub struct ReqwestClient {
    pub client: reqwest::Client,
}

impl ReqwestClient {
    pub fn new(timeout_seconds: Option<u32>)-> Result<
        Self, Box<dyn std::error::Error + Send + Sync>>
    {
        let builder = reqwest::Client::builder();
        let client = if let Some(ts) = timeout_seconds {
            builder.timeout(Duration::from_secs(ts as u64)).build()?
        } else {
            builder.build()?
        };
        Ok(Self { client })
    }
}

pub trait BlockingReqwestClientTrait
{
    fn client_mut(&mut self) -> &mut BlockingReqwestClient;
    fn client(&self) -> &BlockingReqwestClient;
}

pub struct BlockingReqwestClient {
    pub client: reqwest::blocking::Client,
}

impl BlockingReqwestClient {
    pub fn new(timeout_seconds: Option<u32>)-> Result<
        Self, Box<dyn std::error::Error + Send + Sync>>
    {
        let builder = reqwest::blocking::Client::builder();
        let client = if let Some(ts) = timeout_seconds {
            builder.timeout(Duration::from_secs(ts as u64)).build()?
        } else {
            builder.build()?
        };
        Ok(Self { client })
    }
}
