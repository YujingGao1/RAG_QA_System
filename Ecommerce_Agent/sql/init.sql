-- ============================================================
-- 电商客服 Agent — 数据库初始化
-- ============================================================

CREATE DATABASE IF NOT EXISTS ecommerce
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE ecommerce;

-- 订单表
DROP TABLE IF EXISTS orders;
CREATE TABLE orders (
    order_id   INT PRIMARY KEY COMMENT '订单号',
    customer   VARCHAR(50)  NOT NULL COMMENT '客户姓名',
    status     VARCHAR(20)  NOT NULL COMMENT '状态: 待付款/已发货/已完成/已取消',
    items      VARCHAR(200) NOT NULL COMMENT '商品名称',
    total      DECIMAL(10,2) NOT NULL COMMENT '金额',
    created_at DATETIME     NOT NULL COMMENT '下单时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 示例数据
INSERT INTO orders VALUES
(10001, '张三', '已发货',  'iPhone 15 Pro',       8999.00, '2026-07-01 10:30:00'),
(10002, '张三', '待付款',  'MacBook Air',          9499.00, '2026-07-03 14:20:00'),
(10003, '李四', '已完成',  'AirPods Pro',          1899.00, '2026-06-28 09:15:00'),
(10004, '王五', '已发货',  'iPad Air + Apple Pencil', 5299.00, '2026-07-05 16:00:00'),
(10005, '张三', '已取消',  'Apple Watch Ultra',    6999.00, '2026-07-02 08:45:00');
