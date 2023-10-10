// 获取表格元素
var table = document.getElementsByTagName('table')[0];

// 获取表格中的所有行
var rows = table.getElementsByTagName('tr');

// 计算表格总行数和总页数
var rowCount = rows.length - 1; // 减去表头行
var pageCount = Math.ceil(rowCount / 15); // 每页显示20行

// 创建页码链接
var pagination = document.createElement('div');
pagination.className = 'pagination';

for (var i = 1; i <= pageCount; i++) {
  var link = document.createElement('a');
  link.href = '#';
  link.innerHTML = i;
  link.onclick = function() {
    // 获取当前页码
    var currentPage = parseInt(this.innerHTML);

    // 计算当前页的起始行和结束行
    var startRow = (currentPage - 1) * 20 + 1; // 加上表头行
    var endRow = Math.min(currentPage * 20, rowCount) + 1; // 加上表头行

    // 隐藏所有行
    for (var j = 1; j < rows.length; j++) {
      rows[j].style.display = 'none';
    }

    // 显示当前页的行
    for (var j = startRow; j < endRow; j++) {
      rows[j].style.display = '';
    }

    // 更新页码链接的样式
    var links = pagination.getElementsByTagName('a');
    for (var j = 0; j < links.length; j++) {
      links[j].classList.remove('active');
    }
    this.classList.add('active');

    // 阻止链接的默认行为
    return false;
  };

  pagination.appendChild(link);
}

// 将页码链接添加到页面中
table.parentNode.insertBefore(pagination, table.nextSibling);

// 默认显示第一页
pagination.getElementsByTagName('a')[0].click();


function showConfirmation(event) {
    event.preventDefault();
    if (confirm("确定要下载文件吗？")) {
      var url = event.target.href;
      var xhr = new XMLHttpRequest();
      xhr.open('GET', url, true);
      xhr.responseType = 'blob';
      xhr.onload = function() {
        if (this.status === 200) {
          var blob = new Blob([this.response], {type: 'application/octet-stream'});
          var link = document.createElement('a');
          link.href = window.URL.createObjectURL(blob);
          link.download = url.substring(url.lastIndexOf('/')+1);
          link.click();
        }
      };
      xhr.send();
    }
  }