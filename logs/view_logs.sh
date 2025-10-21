#!/bin/bash
# 日志查看和管理脚本

echo "=== IndicTrans2 项目日志管理 ==="
echo "日志目录: logs/slurm_outputs/"
echo ""

# 显示最新的日志文件
echo "📁 最新的日志文件:"
ls -lt logs/slurm_outputs/ | head -10

echo ""
echo "🔍 可用的日志查看命令:"
echo "1. 查看所有日志: ls -la logs/slurm_outputs/"
echo "2. 查看最新日志: tail -f logs/slurm_outputs/slurm-*.out"
echo "3. 查看特定作业日志: cat logs/slurm_outputs/slurm-[作业ID].out"
echo "4. 查看错误日志: cat logs/slurm_outputs/slurm-[作业ID].err"
echo "5. 清理旧日志: rm logs/slurm_outputs/slurm-*.out logs/slurm_outputs/slurm-*.err"
echo ""

# 显示当前运行的任务
echo "🔄 当前运行的任务:"
squeue -M snowy -u maoxuan 2>/dev/null || echo "无法获取任务状态"

echo ""
echo "📊 日志文件统计:"
echo "总日志文件数: $(ls logs/slurm_outputs/ | wc -l)"
echo "输出文件数: $(ls logs/slurm_outputs/*.out 2>/dev/null | wc -l)"
echo "错误文件数: $(ls logs/slurm_outputs/*.err 2>/dev/null | wc -l)"
