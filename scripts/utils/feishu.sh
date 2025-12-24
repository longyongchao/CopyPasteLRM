# # 示例 1: 任务开始通知
# send_feishu_msg "🚀 训练任务启动\n项目: test \n模型: test \n节点: $(hostname)"
# # 示例 3: 任务完成
# send_feishu_msg "✅ 整个 RLHF Pipeline 运行成功！\n输出目录: xxx"
function send_feishu_msg() {
    local msg_content=$1

    local webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/880e2480-71ed-4f29-8495-b7fa75c8cbd7"
    local secret="IzE5LR2O7ojQkRUO9g96Qe"

    if [[ -z "$webhook_url" ]]; then
        echo "[Warn] Lark Webhook URL is not set."
        return 1
    fi

    # 获取当前时间戳
    local timestamp=$(date +%s)
    local sign=""

    # 如果有 Secret，则计算签名
    if [[ -n "$secret" ]]; then
        # 使用 Python 确保计算逻辑与飞书官方要求完全一致
        sign=$(python3 -c "
import hashlib
import hmac
import base64
timestamp = '$timestamp'
secret = '$secret'
string_to_sign = '{}\n{}'.format(timestamp, secret)
hmac_code = hmac.new(string_to_sign.encode('utf-8'), digestmod=hashlib.sha256).digest()
sign = base64.b64encode(hmac_code).decode('utf-8')
print(sign)
")
    fi

    # 构造符合你手册要求的 JSON
    # 如果有签名，加入 timestamp 和 sign；如果没有，按普通格式发送
    local json_data
    if [[ -n "$sign" ]]; then
        json_data=$(cat <<EOF
{
    "timestamp": "$timestamp",
    "sign": "$sign",
    "msg_type": "text",
    "content": {
        "text": "$msg_content"
    }
}
EOF
)
    else
        json_data="{\"msg_type\":\"text\",\"content\":{\"text\":\"$msg_content\"}}"
    fi

    # 发送请求
    curl -s -X POST -H "Content-Type: application/json" \
         -d "$json_data" \
         "$webhook_url"
    echo -e "\n"
}

